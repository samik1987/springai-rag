package com.example.springai_rag.controllers;

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
import org.springframework.ai.rag.retrieval.search.VectorStoreDocumentRetriever;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.core.io.ClassPathResource;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
public class VectorController {

    @Autowired
    private VectorStoreService vectorStoreService;

    @Autowired
    private VectorStore vectorStore;

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
        //PromptTemplate template = PromptTemplate.builder().template("we are act like {role}").build();

        List<Document> documents = this.vectorStoreService.doSimilaritySearch(message);

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



        return _openAiChatClient.prompt()
                .system(renderTemplate)
                .user(message)
                .call()
                .content();




    }


    // RAG with auto Semantic Search by Advisor
    @PostMapping("/openai/chatv2")
    public String getOpenAIResponseMetadataV2(@RequestBody String message, @RequestHeader String userid)
    {

        SearchRequest srchReq = SearchRequest.builder().topK(3)
                .similarityThreshold(0.6)
                .query(message)
                .build();

        QuestionAnswerAdvisor questionAnswerAdvisor =
                QuestionAnswerAdvisor.builder(vectorStore).searchRequest(srchReq).build();

        var retriever = VectorStoreDocumentRetriever.builder()
                .vectorStore(vectorStore)
                .topK(3)
                .similarityThreshold(0.8)
                .build();


        return _openAiChatClient.prompt()
                .advisors(questionAnswerAdvisor)
                //.advisors(RetrievalAugmentationAdvisor.builder().documentRetriever(retriever).build())
                .user(message)
                .call()
                .content();




    }


    // Advance RAG pipeline = Pre + Post
    @PostMapping("/openai/chatv3")
    public String getOpenAIResponseMetadataV3(@RequestBody String message, @RequestHeader String userid) {

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
                                .vectorStore(vectorStore)
                                .topK(5)
                                .similarityThreshold(0.4)
                                .build()
                )

                // 🔹 Join docs
                .documentJoiner(new ConcatenationDocumentJoiner())

                // 🔹 Add context
                .queryAugmenter(
                        ContextualQueryAugmenter.builder().build()
                )

                .build();

        return _openAiChatClient.prompt()
                .advisors(advisor)
                .user(message)
                .call()
                .content();
    }





}
