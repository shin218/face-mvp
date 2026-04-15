package com.example.demo.service.vibe;

/**
 * Face++ 기반 수치를 LLM에 전달하기 위한 입력 DTO.
 */
public class VibeInput {
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
