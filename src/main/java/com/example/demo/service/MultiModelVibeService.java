package com.example.demo.service;

import com.example.demo.service.vibe.ModelVibeResult;
import com.example.demo.service.vibe.VibeInput;
import com.example.demo.service.vibe.VibePrompts;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
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

	public MultiModelVibeService(
			ClaudeVibeService claudeVibeService,
			OpenAiVibeService openAiVibeService,
			GeminiVibeService geminiVibeService
	) {
		this.claudeVibeService = claudeVibeService;
		this.openAiVibeService = openAiVibeService;
		this.geminiVibeService = geminiVibeService;
	}

	/**
	 * 세 모델에 동일 프롬프트로 병렬 호출합니다.
	 */
	public List<ModelVibeResult> analyzeParallel(VibeInput input) {
		if (log.isDebugEnabled()) {
			log.debug("Vibe user prompt (full):\n{}", VibePrompts.buildUserText(input));
		}

		CompletableFuture<ModelVibeResult> claude = CompletableFuture.supplyAsync(() ->
				safeCall("Claude (Sonnet)", () -> {
					if (!claudeVibeService.isConfigured()) {
						throw new IllegalStateException("ANTHROPIC_API_KEY(또는 claude.api.key)가 설정되지 않았습니다.");
					}
					return claudeVibeService.analyzeVibe(input);
				}));
		CompletableFuture<ModelVibeResult> gpt = CompletableFuture.supplyAsync(() ->
				safeCall("GPT-4o", () -> {
					if (!openAiVibeService.isConfigured()) {
						throw new IllegalStateException("OPENAI_API_KEY(또는 openai.api.key)가 설정되지 않았습니다.");
					}
					return openAiVibeService.analyzeVibe(input);
				}));
		CompletableFuture<ModelVibeResult> gemini = CompletableFuture.supplyAsync(() ->
				safeCall("Gemini", () -> {
					if (!geminiVibeService.isConfigured()) {
						throw new IllegalStateException("GEMINI_API_KEY(또는 gemini.api.key)가 설정되지 않았습니다.");
					}
					return geminiVibeService.analyzeVibe(input);
				}));

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
