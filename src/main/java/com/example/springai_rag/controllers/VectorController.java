package com.example.springai_rag.controllers;

import com.example.springai_rag.services.LoggingDocumentReranker;
import com.example.springai_rag.services.VectorStoreService;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.vectorstore.QuestionAnswerAdvisor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.document.Document;
import org.springframework.ai.rag.advisor.RetrievalAugmentationAdvisor;
import org.springframework.ai.rag.generation.augmentation.ContextualQueryAugmenter;
import org.springframework.ai.rag.preretrieval.query.expansion.MultiQueryExpander;
import org.springframework.ai.rag.preretrieval.query.transformation.RewriteQueryTransformer;
import org.springframework.ai.rag.retrieval.join.ConcatenationDocumentJoiner;
import org.springframework.ai.openai.OpenAiChatModel;
import org.springframework.ai.rag.retrieval.search.VectorStoreDocumentRetriever;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.ai.vectorstore.filter.Filter;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.core.io.ClassPathResource;
import org.springframework.web.bind.annotation.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

@RestController
public class VectorController {

    private static final Logger log = LoggerFactory.getLogger(VectorController.class);

    @Autowired
    private VectorStoreService vectorStoreService;

    @Autowired
    private VectorStore vectorStore;

    @Autowired
    private OpenAiChatModel chatModel;

    @Autowired
    @Qualifier("OpenAiChatClient")
    private ChatClient _openAiChatClient;

    @GetMapping("/savedata")
    public void SaveData()
    {
        vectorStoreService.SaveVectorData();
    }

    @GetMapping("/savepdfdata")
    public void SavePadfData()
    {
        vectorStoreService.SavePdfVectorData();
    }

    // RAG with Manual Similarity Search
    @PostMapping("/openai/chatv1")
    public String getOpenAIResponseMetadata(@RequestBody String message, @RequestHeader String userid)
    {
        long started = System.nanoTime();
        //PromptTemplate template = PromptTemplate.builder().template("we are act like {role}").build();

        long searchStarted = System.nanoTime();
        List<Document> documents = this.vectorStoreService.doSimilaritySearch(message);
        long vectorSearchMs = millisSince(searchStarted);

        // Extra to see the similarity search result
        List<String> documentList = documents.stream().map(Document::getText).toList();

        String filterContext = String.join(" , ",documentList);

        System.out.println("Similarity Search Result = "+filterContext);
        // Extra to see the similarity search result

        PromptTemplate template = new PromptTemplate(
                new ClassPathResource("prompts/user-template.st")
        );

        String renderTemplate = template.render(Map.of(
                "role","Software Engineer",
                "audience-role","School Students",
                "context", filterContext
        ));



        long llmStarted = System.nanoTime();
        String answer = _openAiChatClient.prompt()
                .advisors(spec -> spec.param(ChatMemory.CONVERSATION_ID, userid))
                .system(renderTemplate)
                .user(message)
                .call()
                .content();
        logTiming("chatv1", millisSince(started), vectorSearchMs, millisSince(llmStarted));
        return answer;
    }


    // RAG with auto Semantic Search by Advisor
    @PostMapping("/openai/chatv2")
    public String getOpenAIResponseMetadataV2(@RequestBody String message, @RequestHeader String userid)
    {
        long started = System.nanoTime();
        TimingVectorStore timedStore = new TimingVectorStore(vectorStore);

        SearchRequest srchReq = SearchRequest.builder().topK(3)
                .similarityThreshold(0.6)
                .query(message)
                .build();

        QuestionAnswerAdvisor questionAnswerAdvisor =
                QuestionAnswerAdvisor.builder(timedStore).searchRequest(srchReq).build();

        var retriever = VectorStoreDocumentRetriever.builder()
                .vectorStore(vectorStore)
                .topK(3)
                .similarityThreshold(0.8)
                .build();


        long callStarted = System.nanoTime();
        String answer = _openAiChatClient.prompt()
                .advisors(spec -> spec.param(ChatMemory.CONVERSATION_ID, userid))
                .advisors(questionAnswerAdvisor)
                //.advisors(RetrievalAugmentationAdvisor.builder().documentRetriever(retriever).build())
                .user(message)
                .call()
                .content();
        long callMs = millisSince(callStarted);
        logTiming("chatv2", millisSince(started), timedStore.searchMillis(), callMs - timedStore.searchMillis());
        return answer;
    }


    // Advance RAG pipeline = Pre + Post
    @PostMapping("/openai/chatv3")
    public String getOpenAIResponseMetadataV3(@RequestBody String message, @RequestHeader String userid) {
        long started = System.nanoTime();
        TimingVectorStore timedStore = new TimingVectorStore(vectorStore);

        var advisor = RetrievalAugmentationAdvisor.builder()

                // 🔹 Query Rewrite
                .queryTransformers(
                        RewriteQueryTransformer.builder()
                                .chatClientBuilder(_openAiChatClient.mutate().clone())
                                .build()
                )

                // 🔹 Query Expansion
                .queryExpander(
                        MultiQueryExpander.builder()
                                .chatClientBuilder(_openAiChatClient.mutate().clone())
                                .build()
                )

                // 🔹 Retrieval
                .documentRetriever(
                        VectorStoreDocumentRetriever.builder()
                                .vectorStore(timedStore)
                                .topK(8)
                                .similarityThreshold(0.4)
                                .build()
                )

                // 🔹 Join docs
                .documentJoiner(new ConcatenationDocumentJoiner())

                // 🔹 Rerank the retrieved chunks and keep the best 3
                .documentPostProcessors(new LoggingDocumentReranker(ChatClient.builder(chatModel).build()))

                // 🔹 Add context
                .queryAugmenter(
                        ContextualQueryAugmenter.builder().build()
                )

                .build();

        long callStarted = System.nanoTime();
        String answer = _openAiChatClient.prompt()
                .advisors(spec -> spec.param(ChatMemory.CONVERSATION_ID, userid))
                .advisors(advisor)
                .user(message)
                .call()
                .content();
        long callMs = millisSince(callStarted);
        logTiming("chatv3", millisSince(started), timedStore.searchMillis(), callMs - timedStore.searchMillis());
        return answer;
    }

    private long millisSince(long startNanos) {
        return TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - startNanos);
    }

    private void logTiming(String endpoint, long totalMs, long vectorSearchMs, long llmMs) {
        log.info("{} total={} ms, vectorSearch={} ms, llm={} ms", endpoint, totalMs, vectorSearchMs, llmMs);
    }

    private static final class TimingVectorStore implements VectorStore {

        private final VectorStore delegate;
        private long searchNanos;

        private TimingVectorStore(VectorStore delegate) {
            this.delegate = delegate;
        }

        private long searchMillis() {
            return TimeUnit.NANOSECONDS.toMillis(searchNanos);
        }

        @Override
        public List<Document> similaritySearch(SearchRequest request) {
            long started = System.nanoTime();
            try {
                return delegate.similaritySearch(request);
            }
            finally {
                searchNanos += System.nanoTime() - started;
            }
        }

        @Override
        public void add(List<Document> documents) {
            delegate.add(documents);
        }

        @Override
        public void delete(List<String> idList) {
            delegate.delete(idList);
        }

        @Override
        public void delete(Filter.Expression filterExpression) {
            delegate.delete(filterExpression);
        }
    }





}
