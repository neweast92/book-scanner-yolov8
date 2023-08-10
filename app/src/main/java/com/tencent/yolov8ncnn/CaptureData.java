package com.tencent.yolov8ncnn;

import android.graphics.Bitmap;
import android.graphics.PointF;
import android.os.Parcel;
import android.os.Parcelable;


public class CaptureData implements Parcelable {
    public Bitmap img;
    public PointF[] vtx = new PointF[4];
    public float score = 0.0f;

    public CaptureData(){

    }

    protected CaptureData(Parcel in) {
        img = in.readParcelable(Bitmap.class.getClassLoader());
        vtx = in.createTypedArray(PointF.CREATOR);
        score = in.readFloat();
    }

    public static final Creator<CaptureData> CREATOR = new Creator<CaptureData>() {
        @Override
        public CaptureData createFromParcel(Parcel in) {
            return new CaptureData(in);
        }

        @Override
        public CaptureData[] newArray(int size) {
            return new CaptureData[size];
        }
    };

    @Override
    public int describeContents() {
        return 0;
    }

    @Override
    public void writeToParcel(Parcel parcel, int i) {
        parcel.writeParcelable(img, i);
        parcel.writeTypedArray(vtx, i);
        parcel.writeFloat(score);
    }
}
