package com.example.demo.service.vibe;

/**
 * Face++ 기반 수치 + 고객 사진/레퍼런스/텍스트를 LLM에 전달하기 위한 입력 DTO.
 */
public class VibeInput {
	// 이미지 데이터 (Claude 전용)
	public byte[] customerImageBytes;
	public String customerImageMime;
	public byte[] ref1Bytes;
	public String ref1Mime;
	public byte[] ref2Bytes;
	public String ref2Mime;
	public String vibeText;

	public String detectPrettyJson;

	public String gender;
	public Integer age;
	public String emotionSummary;

	public double skinHealth;
	public double skinStain;
	public double skinAcne;
	public double skinDarkCircle;

	public Double smile;
	public String glass;

	public Double eyeDistancePx;
	public Double faceWidthHeightRatio;
	public Double leftEyeWidthPx;
	public Double rightEyeWidthPx;
	public Double noseChinToForeheadNoseRatio;
	public Double lipThicknessPx;
	public String faceShape;
	public Boolean bangsLikely;
}
