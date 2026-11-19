package com.example.demo.service;

import com.example.demo.service.vibe.ModelVibeResult;
import com.example.demo.service.vibe.VibeInput;
import com.example.demo.service.vibe.VibePrompts;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.function.Supplier;

@Service
public class MultiModelVibeService {

	private static final Logger log = LoggerFactory.getLogger(MultiModelVibeService.class);

	private final ClaudeVibeService claudeVibeService;
	private final OpenAiVibeService openAiVibeService;
	private final GeminiVibeService geminiVibeService;

	private final boolean gptEnabled;
	private final boolean geminiEnabled;

	public MultiModelVibeService(
			ClaudeVibeService claudeVibeService,
			OpenAiVibeService openAiVibeService,
			GeminiVibeService geminiVibeService,
			@Value("${features.gpt-enabled:false}") boolean gptEnabled,
			@Value("${features.gemini-enabled:false}") boolean geminiEnabled
	) {
		this.claudeVibeService = claudeVibeService;
		this.openAiVibeService = openAiVibeService;
		this.geminiVibeService = geminiVibeService;
		this.gptEnabled = gptEnabled;
		this.geminiEnabled = geminiEnabled;
	}

	/**
	 * Claude는 항상 호출. GPT/Gemini는 feature flag가 활성화된 경우에만 호출.
	 */
	public List<ModelVibeResult> analyzeParallel(VibeInput input) {
		if (log.isDebugEnabled()) {
			log.debug("Vibe analysis start — gptEnabled={}, geminiEnabled={}", gptEnabled, geminiEnabled);
		}

		CompletableFuture<ModelVibeResult> claude = CompletableFuture.supplyAsync(() ->
				safeCall("Claude (Sonnet)", () -> {
					if (!claudeVibeService.isConfigured()) {
						throw new IllegalStateException("ANTHROPIC_API_KEY(또는 claude.api.key)가 설정되지 않았습니다.");
					}
					return claudeVibeService.analyzeVibe(input);
				}));

		CompletableFuture<ModelVibeResult> gpt = gptEnabled
				? CompletableFuture.supplyAsync(() ->
						safeCall("GPT-4o", () -> {
							if (!openAiVibeService.isConfigured()) {
								throw new IllegalStateException("OPENAI_API_KEY(또는 openai.api.key)가 설정되지 않았습니다.");
							}
							return openAiVibeService.analyzeVibe(input);
						}))
				: CompletableFuture.completedFuture(new ModelVibeResult("GPT-4o", "", "비활성화됨 (features.gpt-enabled=false)"));

		CompletableFuture<ModelVibeResult> gemini = geminiEnabled
				? CompletableFuture.supplyAsync(() ->
						safeCall("Gemini", () -> {
							if (!geminiVibeService.isConfigured()) {
								throw new IllegalStateException("GEMINI_API_KEY(또는 gemini.api.key)가 설정되지 않았습니다.");
							}
							return geminiVibeService.analyzeVibe(input);
						}))
				: CompletableFuture.completedFuture(new ModelVibeResult("Gemini", "", "비활성화됨 (features.gemini-enabled=false)"));

		CompletableFuture.allOf(claude, gpt, gemini).join();
		List<ModelVibeResult> ordered = new ArrayList<>(3);
		ordered.add(claude.join());
		ordered.add(gpt.join());
		ordered.add(gemini.join());
		return ordered;
	}

	private static ModelVibeResult safeCall(String provider, Supplier<String> call) {
		try {
			String text = call.get();
			return new ModelVibeResult(provider, text, null);
		} catch (Exception e) {
			String msg = e.getMessage() == null ? e.getClass().getSimpleName() : e.getMessage();
			return new ModelVibeResult(provider, "", msg);
		}
	}
}
