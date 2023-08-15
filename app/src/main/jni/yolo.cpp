// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "yolo.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cpu.h"

#define MAX_STRIDE 32

static void slice(const ncnn::Mat& in, ncnn::Mat& out, int start, int end, int axis)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Crop");

    // set param
    ncnn::ParamDict pd;

    ncnn::Mat axes = ncnn::Mat(1);
    axes.fill(axis);
    ncnn::Mat ends = ncnn::Mat(1);
    ends.fill(end);
    ncnn::Mat starts = ncnn::Mat(1);
    starts.fill(start);
    pd.set(9, starts);// start
    pd.set(10, ends);// end
    pd.set(11, axes);//axes

    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    op->forward(in, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}
static void interp(const ncnn::Mat& in, const float& scale, const int& out_w, const int& out_h, ncnn::Mat& out)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Interp");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, 2);// resize_type
    pd.set(1, scale);// height_scale
    pd.set(2, scale);// width_scale
    pd.set(3, out_h);// height
    pd.set(4, out_w);// width

    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    op->forward(in, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}
static void reshape(const ncnn::Mat& in, ncnn::Mat& out, int c, int h, int w, int d)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Reshape");

    // set param
    ncnn::ParamDict pd;

    pd.set(0, w);// start
    pd.set(1, h);// end
    if (d > 0)
        pd.set(11, d);//axes
    pd.set(2, c);//axes
    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    op->forward(in, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}
static void sigmoid(ncnn::Mat& bottom)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Sigmoid");

    op->create_pipeline(opt);

    // forward

    op->forward_inplace(bottom, opt);
    op->destroy_pipeline(opt);

    delete op;
}
static float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}
static float sigmoid(float x)
{
    return 1.0f / (1.0f + fast_exp(-x));
}

static void matmul(const std::vector<ncnn::Mat>& bottom_blobs, ncnn::Mat& top_blob)
{
    ncnn::Option opt;
    opt.num_threads = 2;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("MatMul");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, 0);// axis

    op->load_param(pd);

    op->create_pipeline(opt);
    std::vector<ncnn::Mat> top_blobs(1);
    op->forward(bottom_blobs, top_blobs, opt);
    top_blob = top_blobs[0];

    op->destroy_pipeline(opt);

    delete op;
}
static float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    //     #pragma omp parallel sections
    {
        //         #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        //         #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.width * faceobjects[i].rect.height;
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold or i != 0)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static void generate_proposals(std::vector<GridAndStride> grid_strides, const ncnn::Mat& pred, float prob_threshold, std::vector<Object>& objects)
{
    const int num_points = grid_strides.size();
    const int num_class = 1;
    const int reg_max_1 = 16;

    for (int i = 0; i < num_points; i++)
    {
        const float* scores = pred.row(i) + 4 * reg_max_1;

        // find label with max score
        int label = -1;
        float score = -FLT_MAX;
        for (int k = 0; k < num_class; k++)
        {
            float confidence = scores[k];
            if (confidence > score)
            {
                label = k;
                score = confidence;
            }
        }
        float box_prob = sigmoid(score);
        if (box_prob >= prob_threshold)
        {
            ncnn::Mat bbox_pred(reg_max_1, 4, (void*)pred.row(i));
            {
                ncnn::Layer* softmax = ncnn::create_layer("Softmax");

                ncnn::ParamDict pd;
                pd.set(0, 1); // axis
                pd.set(1, 1);
                softmax->load_param(pd);

                ncnn::Option opt;
                opt.num_threads = 1;
                opt.use_packing_layout = false;

                softmax->create_pipeline(opt);

                softmax->forward_inplace(bbox_pred, opt);

                softmax->destroy_pipeline(opt);

                delete softmax;
            }

            float pred_ltrb[4];
            for (int k = 0; k < 4; k++)
            {
                float dis = 0.f;
                const float* dis_after_sm = bbox_pred.row(k);
                for (int l = 0; l < reg_max_1; l++)
                {
                    dis += l * dis_after_sm[l];
                }

                pred_ltrb[k] = dis * grid_strides[i].stride;
            }

            float pb_cx = (grid_strides[i].grid0 + 0.5f) * grid_strides[i].stride;
            float pb_cy = (grid_strides[i].grid1 + 0.5f) * grid_strides[i].stride;

            float x0 = pb_cx - pred_ltrb[0];
            float y0 = pb_cy - pred_ltrb[1];
            float x1 = pb_cx + pred_ltrb[2];
            float y1 = pb_cy + pred_ltrb[3];

            Object obj;
            obj.rect.x = x0;
            obj.rect.y = y0;
            obj.rect.width = x1 - x0;
            obj.rect.height = y1 - y0;
            obj.label = label;
            obj.prob = box_prob;
            obj.mask_feat.resize(32);
            std::copy(pred.row(i) + 64 + num_class, pred.row(i) + 64 + num_class + 32, obj.mask_feat.begin());
            objects.push_back(obj);
        }
    }
}
static void generate_grids_and_stride(const int target_w, const int target_h, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
{
    for (int i = 0; i < (int)strides.size(); i++)
    {
        int stride = strides[i];
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;
        for (int g1 = 0; g1 < num_grid_h; g1++)
        {
            for (int g0 = 0; g0 < num_grid_w; g0++)
            {
                GridAndStride gs;
                gs.grid0 = g0;
                gs.grid1 = g1;
                gs.stride = stride;
                grid_strides.push_back(gs);
            }
        }
    }
}

/*
     * mask_feat: 객체의 마스크 특징
     * img_w, img_h: 원본 이미지의 너비와 높이
     * in_pad: 패딩된 입력 이미지
     * mask_proto: 출력 마스크
     * wpad, hpad: 너비와 높이에 추가된 패딩값
     * mask_pred_result: 디코딩된 마스크 결과를 저장할 Mat
     * */
static void decode_mask(const ncnn::Mat& mask_feat, const int& img_w, const int& img_h,
                        const ncnn::Mat& mask_proto, const ncnn::Mat& in_pad, const int& wpad, const int& hpad,
                        ncnn::Mat& mask_pred_result)
{
    ncnn::Mat masks;
    matmul(std::vector<ncnn::Mat>{mask_feat, mask_proto}, masks); // 행렬곱하여 masks에 저장
    sigmoid(masks); // 각 요소에 시그모이드 함수 적용
    reshape(masks, masks, masks.h, in_pad.h / 4, in_pad.w / 4, 0); // masks 모양을 변경하여 해당 마스크의 높이와 패딩된 입력의 너비와 높이의 1/4로 설정
    slice(masks, mask_pred_result, (wpad / 2) / 4, (in_pad.w - wpad / 2) / 4, 2); // masks의 너비를 wpad/2 부터 in_pad.w - wpad/2 사이로 슬라이스하여 mask_pred_result에 저장
    slice(mask_pred_result, mask_pred_result, (hpad / 2) / 4, (in_pad.h - hpad / 2) / 4, 1); // mask_pred_result의 높이를 hpad/2 부터 in_pad.h - hpad/2 사이로 슬라이스하여 다시 저장
    interp(mask_pred_result, 4.0, img_w, img_h, mask_pred_result); // mask_pred_result의 크기를 조정하여 원본 이미지 크기에 맞게 변경

}


// Calculate slope and y-intercept using Least Squares Method
float calculateSlope(const std::vector<cv::Point2i>& points) {
    int n = points.size();
    if(n == 0) return 0;

    float sum_x = 0, sum_y = 0, sum_x2 = 0, sum_xy = 0;
    for(const auto& point : points) {
        sum_x += point.x;
        sum_y += point.y;
        sum_x2 += point.x * point.x;
        sum_xy += point.x * point.y;
    }
    float m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    return m;
}

float calculateIntercept(const std::vector<cv::Point2i>& points, float slope) {
    if(points.size() == 0) return 0;

    float mean_y = 0, mean_x = 0;
    for(const auto& point : points) {
        mean_x += point.x;
        mean_y += point.y;
    }
    mean_x /= points.size();
    mean_y /= points.size();

    return mean_y - slope * mean_x;
}

cv::Point2f findIntersection(float m1, float c1, float m2, float c2) {
    float x = (c2 - c1) / (m1 - m2);
    float y = m1 * x + c1;
    return cv::Point2f(x, y);
}


Yolo::Yolo()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int Yolo::load(const char* modeltype, int _target_size, const float* _mean_vals,  const float* _norm_vals, bool use_gpu)
{
    yolo.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    yolo.opt = ncnn::Option();

#if NCNN_VULKAN
    yolo.opt.use_vulkan_compute = use_gpu;
#endif

    yolo.opt.num_threads = ncnn::get_big_cpu_count();
    yolo.opt.blob_allocator = &blob_pool_allocator;
    yolo.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "yolov8%s-seg.param", modeltype);
    sprintf(modelpath, "yolov8%s-seg.bin", modeltype);

    yolo.load_param(parampath);
    yolo.load_model(modelpath);


    target_size = _target_size;
    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}

int Yolo::load(AAssetManager* mgr, const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    // 각각의 객체와 할당자를 초기 상태로 리셋
    yolo.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2); // CPU 파워 세이브 모드를 설정. 2 -> 최대성능모드
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count()); // OpenMP로 사용할 스레드 개수를 big CPU 코어수만큼 설정

    yolo.opt = ncnn::Option();

#if NCNN_VULKAN
    yolo.opt.use_vulkan_compute = use_gpu; // GPU 사용한 연산 수행 여부
#endif

    yolo.opt.num_threads = ncnn::get_big_cpu_count();
    yolo.opt.blob_allocator = &blob_pool_allocator;
    yolo.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "yolov8%s-seg.param", modeltype);
    sprintf(modelpath, "yolov8%s-seg.bin", modeltype);

    yolo.load_param(mgr, parampath);
    yolo.load_model(mgr, modelpath);

    target_size = _target_size; // 대상 이미지 크기
    mean_vals[0] = _mean_vals[0]; // 이미지 전처리 위한 평균값
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0]; // 정규화값
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}

// prob_threshold: 확률 임계값. 이보다 낮은 객체는 무시
int Yolo::detect(const cv::Mat& rgb, std::vector<Object>& objects, float prob_threshold, float nms_threshold)
{
    // 이미지의 가로 세로 크기
    int width = rgb.cols;
    int height = rgb.rows;

    // pad to multiple of 32
    int w = width;
    int h = height;
    float scale = 1.f;
    // 가로 세로 중 긴 쪽을 target_size에 맞추고, 다른 쪽은 그에 따라 비율에 맞게 조절
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    //입력 이미지를 지정된 크기 w, h로 크기 조절하고, BGR 형식으로 변환
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB2BGR, width, height, w, h);

    // pad to target_size rectangle
    // 이미지를 MAX_STRIDE의 배수로 패딩하여 직사각형 모양으로 만듦
    int wpad = (w + MAX_STRIDE-1) / MAX_STRIDE * MAX_STRIDE - w;
    int hpad = (h + MAX_STRIDE-1) / MAX_STRIDE * MAX_STRIDE - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);

    // 정규화
    in_pad.substract_mean_normalize(0, norm_vals);

    // 추론
    ncnn::Extractor ex = yolo.create_extractor();

    ex.input("images", in_pad);

    ncnn::Mat out;
    ex.extract("output0", out);

    ncnn::Mat mask_proto;
    ex.extract("output1", mask_proto);

    // 그리드와 스트라이드 생성
    std::vector<int> strides = {8, 16, 32}; // might have stride=64
    std::vector<GridAndStride> grid_strides;
    generate_grids_and_stride(in_pad.w, in_pad.h, strides, grid_strides);

    // 출력에서 객체 제안을 생성
    std::vector<Object> proposals;
    std::vector<Object> objects8;
    generate_proposals(grid_strides, out, prob_threshold, objects8);
    proposals.insert(proposals.end(), objects8.begin(), objects8.end());

    // sort all proposals by score from highest to lowest
    // 점수로 내림차순 정렬
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    // NMS 적용
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();

    ncnn::Mat mask_feat = ncnn::Mat(32, count, sizeof(float));
    for (int i = 0; i < count; i++) {
        float* mask_feat_ptr = mask_feat.row(i);
        std::memcpy(mask_feat_ptr, proposals[picked[i]].mask_feat.data(), sizeof(float) * proposals[picked[i]].mask_feat.size());
    }

    // 마스크 정보를 디코딩하여 원래 이미지 크기에 맞게 크기 조절
    ncnn::Mat mask_pred_result;
    decode_mask(mask_feat, width, height, mask_proto, in_pad, wpad, hpad, mask_pred_result);

    // objects 벡터에 최종 탐지된 객체의 위치와 크기를 원래 이미지 크기로 저장, 클리핑하여 경계를 넘어가지 않도록 하여 저장
    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;

        objects[i].mask = cv::Mat::zeros(height, width, CV_32FC1);
        cv::Mat mask = cv::Mat(height, width, CV_32FC1, (float*)mask_pred_result.channel(i));
        mask(objects[i].rect).copyTo(objects[i].mask(objects[i].rect));
    }

    return 0;
}

int Yolo::draw(cv::Mat& rgb, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
            "book", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
    };
    static const unsigned char colors[81][3] = {
            {56,  0,   255},
            {226, 255, 0},
            {0,   94,  255},
            {0,   37,  255},
            {0,   255, 94},
            {255, 226, 0},
            {0,   18,  255},
            {255, 151, 0},
            {170, 0,   255},
            {0,   255, 56},
            {255, 0,   75},
            {0,   75,  255},
            {0,   255, 169},
            {255, 0,   207},
            {75,  255, 0},
            {207, 0,   255},
            {37,  0,   255},
            {0,   207, 255},
            {94,  0,   255},
            {0,   255, 113},
            {255, 18,  0},
            {255, 0,   56},
            {18,  0,   255},
            {0,   255, 226},
            {170, 255, 0},
            {255, 0,   245},
            {151, 255, 0},
            {132, 255, 0},
            {75,  0,   255},
            {151, 0,   255},
            {0,   151, 255},
            {132, 0,   255},
            {0,   255, 245},
            {255, 132, 0},
            {226, 0,   255},
            {255, 37,  0},
            {207, 255, 0},
            {0,   255, 207},
            {94,  255, 0},
            {0,   226, 255},
            {56,  255, 0},
            {255, 94,  0},
            {255, 113, 0},
            {0,   132, 255},
            {255, 0,   132},
            {255, 170, 0},
            {255, 0,   188},
            {113, 255, 0},
            {245, 0,   255},
            {113, 0,   255},
            {255, 188, 0},
            {0,   113, 255},
            {255, 0,   0},
            {0,   56,  255},
            {255, 0,   113},
            {0,   255, 188},
            {255, 0,   94},
            {255, 0,   18},
            {18,  255, 0},
            {0,   255, 132},
            {0,   188, 255},
            {0,   245, 255},
            {0,   169, 255},
            {37,  255, 0},
            {255, 0,   151},
            {188, 0,   255},
            {0,   255, 37},
            {0,   255, 0},
            {255, 0,   170},
            {255, 0,   37},
            {255, 75,  0},
            {0,   0,   255},
            {255, 207, 0},
            {255, 0,   226},
            {255, 245, 0},
            {188, 255, 0},
            {0,   255, 18},
            {0,   255, 75},
            {0,   255, 151},
            {255, 56,  0},
            {245, 255, 0}
    };
    int color_index = 0;
    // 객체 수만큼 루프
    for (size_t i = 0; i < objects.size(); i++)
    {
        // 현재 객체 정보 obj 가져오고, 색상 color 설정
        const Object& obj = objects[i];
        const unsigned char* color = colors[color_index % 80];
        color_index++;

        cv::Scalar cc(color[0], color[1], color[2]);

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        // 추가
        std::vector<cv::Point2i> upPoints, rightPoints, leftPoints, downPoints;

        cv::Mat mask_rgb(rgb.size(), CV_8UC1, cv::Scalar(0));

        // 이미지의 모든 픽셀 루프
        for (int y = 0; y < rgb.rows; y++) {
            uchar* image_ptr = rgb.ptr(y);
            const float* mask_ptr = obj.mask.ptr<float>(y);

            for (int x = 0; x < rgb.cols; x++) {
                // 마스크의 값이 0.XX 이상이면 해당 픽셀에 색상 적용
                if (mask_ptr[x] >= 0.65)
                {
                    image_ptr[0] = cv::saturate_cast<uchar>(image_ptr[0] * 0.5 + color[2] * 0.5);
                    image_ptr[1] = cv::saturate_cast<uchar>(image_ptr[1] * 0.5 + color[1] * 0.5);
                    image_ptr[2] = cv::saturate_cast<uchar>(image_ptr[2] * 0.5 + color[0] * 0.5);


                    mask_rgb.at<uchar>(y,x) = 255; // 마스크의 이진화 이미지 생성


                    // Find edges and classify the direction
//                    if (y > 0 && y < rgb.rows - 1) {
//                        if (obj.mask.at<float>(y-1, x) < 0.65 && obj.mask.at<float>(y+1, x) >= 0.65) {
//                            upPoints.push_back(cv::Point2i(x,y));
//                        } else if (obj.mask.at<float>(y+1, x) < 0.65 && obj.mask.at<float>(y-1, x) >= 0.65) {
//                            downPoints.push_back(cv::Point2i(x,y));
//                        }
//                    }
//
//                    if (x > 0 && x < rgb.cols - 1) {
//                        if (obj.mask.at<float>(y, x+1) < 0.65 && obj.mask.at<float>(y, x-1) >= 0.65) {
//                            rightPoints.push_back(cv::Point2i(x,y));
//                        } else if (obj.mask.at<float>(y, x-1) < 0.65 && obj.mask.at<float>(y, x+1) >= 0.65) {
//                            leftPoints.push_back(cv::Point2i(x,y));
//                        }
//                    }

                    if (y > 0 && obj.mask.at<float>(y - 1, x) < 0.65) upPoints.emplace_back(x, y);
                    if (y < rgb.rows - 1 && obj.mask.at<float>(y + 1, x) < 0.65) downPoints.emplace_back(x, y);
                    if (x > 0 && obj.mask.at<float>(y, x - 1) < 0.65) leftPoints.emplace_back(x, y);
                    if (x < rgb.cols - 1 && obj.mask.at<float>(y, x + 1) < 0.65) rightPoints.emplace_back(x, y);
                }
                image_ptr += 3;
            }
        }

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 10)); // 5x5 rectangle

        // Apply morphological opening operation
        cv::morphologyEx(mask_rgb, mask_rgb, cv::MORPH_OPEN, kernel);

        // 1. Canny edge detection
        cv::Mat edges;
        cv::Canny(mask_rgb, edges, 50, 150);

        // 2. Find contours
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(edges, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // 3. Approximate contours to find rectangle vertices
        for (size_t i = 0; i < contours.size(); i++) {
            std::vector<cv::Point> hull;
            cv::convexHull(contours[i], hull);
            std::vector<cv::Point> approx;
            cv::approxPolyDP(hull, approx, cv::arcLength(hull, true)*0.02, true);
            if (approx.size() == 4) {
                // 4. Draw the rectangle
                for (int j = 0; j < 4; j++) {
                    cv::line(rgb, approx[j], approx[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
                }
            }
        }




        //추가
//        float upSlope = calculateSlope(upPoints);
//        float upIntercept = calculateIntercept(upPoints, upSlope);
//
//        float downSlope = calculateSlope(downPoints);
//        float downIntercept = calculateIntercept(downPoints, downSlope);
//
//        float leftSlope = calculateSlope(leftPoints);
//        float leftIntercept = calculateIntercept(leftPoints, leftSlope);
//
//        float rightSlope = calculateSlope(rightPoints);
//        float rightIntercept = calculateIntercept(rightPoints, rightSlope);
//
//        cv::Point2f upLeftIntersection = findIntersection(upSlope, upIntercept, leftSlope, leftIntercept);
//        cv::Point2f upRightIntersection = findIntersection(upSlope, upIntercept, rightSlope, rightIntercept);
//        cv::Point2f downLeftIntersection = findIntersection(downSlope, downIntercept, leftSlope, leftIntercept);
//        cv::Point2f downRightIntersection = findIntersection(downSlope, downIntercept, rightSlope, rightIntercept);
//
//        cv::Scalar topPointColor(0, 0, 255); // Red color in BGR format
//        cv::Scalar bottomPointColor(0, 255, 255);
//        cv::Scalar leftPointColor(0, 255, 0);
//        cv::Scalar rightPointColor(255, 0, 0);
//        int pointRadius = 2;
//        int pointThickness = -1; // -1 means the circle is filled
//
//        // 상단 점들 그리기
//        for(const cv::Point2i& pt : upPoints) {
//            cv::circle(rgb, pt, pointRadius, topPointColor, pointThickness);
//        }
//
//        // 우측 점들 그리기
//        for(const cv::Point2i& pt : rightPoints) {
//            cv::circle(rgb, pt, pointRadius, rightPointColor, pointThickness);
//        }
//
//        // 좌측 점들 그리기
//        for(const cv::Point2i& pt : leftPoints) {
//            cv::circle(rgb, pt, pointRadius, leftPointColor, pointThickness);
//        }
//
//        // 하단 점들 그리기
//        for(const cv::Point2i& pt : downPoints) {
//            cv::circle(rgb, pt, pointRadius, bottomPointColor, pointThickness);
//        }
//
//
//        // 직선 그리기
//        cv::Scalar lineColor(0, 255, 0); // Green color in BGR format
//        int thickness = 2;
//
//        // 상단 직선 그리기
//        cv::Point topStart(0, upSlope * 0 + upIntercept);
//        cv::Point topEnd(rgb.cols, upSlope * rgb.cols + upIntercept);
//        cv::line(rgb, topStart, topEnd, lineColor, thickness);
//
//        // 하단 직선 그리기
//        cv::Point bottomStart(0, downSlope * 0 + downIntercept);
//        cv::Point bottomEnd(rgb.cols, downSlope * rgb.cols + downIntercept);
//        cv::line(rgb, bottomStart, bottomEnd, lineColor, thickness);
//
//        // 왼쪽 직선 그리기
//        cv::Point leftStart(0, leftSlope * 0 + leftIntercept);
//        cv::Point leftEnd(rgb.cols, leftSlope * rgb.cols + leftIntercept);
//        cv::line(rgb, leftStart, leftEnd, lineColor, thickness);
//
//        // 오른쪽 직선 그리기
//        cv::Point rightStart(0, rightSlope * 0 + rightIntercept);
//        cv::Point rightEnd(rgb.cols, rightSlope * rgb.cols + rightIntercept);
//        cv::line(rgb, rightStart, rightEnd, lineColor, thickness);
//
//
//
//        // 원의 반경을 설정합니다.
//        int radius = 5;
//        cv::Scalar red_color(0, 0, 255); // Red color in BGR format
//
//        // 교차점들에 점을 찍습니다.
//        cv::circle(rgb, upLeftIntersection, radius, red_color, -1); // -1 means filled circle
//        cv::circle(rgb, upRightIntersection, radius, red_color, -1);
//        cv::circle(rgb, downLeftIntersection, radius, red_color, -1);
//        cv::circle(rgb, downRightIntersection, radius, red_color, -1);





        // 탐지된 객체 주위에 바운딩 박스 그리기
        cv::rectangle(rgb, obj.rect, cc, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        // 텍스트 크기 가져옴
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;

        // 텍스트 레이블을 그릴 백그라운드 박스를 그림
        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        // 객체의 이름과 확률을 그림
        cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    return 0;
}
