package com.example.demo.service.vibe;

public class ModelAfterResult {
	private final String provider;
	private final String imageUrl;
	private final String prompt;
	private final String error;

	public ModelAfterResult(String provider, String imageUrl, String prompt, String error) {
		this.provider = provider;
		this.imageUrl = imageUrl;
		this.prompt = prompt;
		this.error = error;
	}

	public String getProvider() { return provider; }
	public String getImageUrl() { return imageUrl; }
	public String getPrompt() { return prompt; }
	public String getError() { return error; }
	public boolean hasError() { return error != null && !error.isBlank(); }
}
