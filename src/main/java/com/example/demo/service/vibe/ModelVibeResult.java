package com.example.demo.service.vibe;

/**
 * 단일 LLM의 바이브 분석 결과.
 */
public class ModelVibeResult {
	private final String provider;
	private final String text;
	private final String error;

	public ModelVibeResult(String provider, String text, String error) {
		this.provider = provider;
		this.text = text;
		this.error = error;
	}

	public String getProvider() {
		return provider;
	}

	public String getText() {
		return text;
	}

	public String getError() {
		return error;
	}

	public boolean hasError() {
		return error != null && !error.isBlank();
	}
}
