package com.example.springai_rag.services;

import com.example.springai_rag.db.ListData;
import com.example.springai_rag.db.PdfData;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.jdbc.core.BatchPreparedStatementSetter;
import org.springframework.jdbc.core.JdbcTemplate;

import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

public class VectorStoreService {

    @Autowired
    private VectorStore vectorStore;

    @Autowired
    private PdfData pdfData;

    @Autowired
    private EmbeddingModel embeddingClient;

    @Autowired
    private JdbcTemplate jdbcTemplate;

    @Value("${spring.ai.vectorstore.mariadb.table-name}")
    private String vectorTableName;

    @Value("${spring.ai.openai.embedding.options.model}")
    private String embeddingModel;

    @Value("${spring.ai.vectorstore.mariadb.dimensions}")
    private int embeddingDimensions;

    public void SaveVectorData()
    {
        System.out.println("Vector Data saving started..");
        List<String> texts = ListData.getData();
        List<Document> documents = new ArrayList<>();
        for (int index = 0; index < texts.size(); index++) {
            Map<String, Object> metadata = embeddingMetadata("list-data");
            metadata.put("index", index);
            documents.add(new Document(UUID.randomUUID().toString(), texts.get(index), metadata));
        }

        deleteBySource("list-data");
        jdbcTemplate.update("DELETE FROM " + vectorTable() + " WHERE metadata = '{}' OR metadata IS NULL");
        vectorStore.add(documents);
        storeVectorInMetadata(documents);
        System.out.println("Data saving Completed");
    }

    public  List<Document> doSimilaritySearch(String msg)
    {
        SearchRequest srchReq = SearchRequest.builder().topK(3)
                .similarityThreshold(0.6)
                .query(msg)
                .build();

        return vectorStore.similaritySearch(srchReq);
    }

    public  void SavePdfVectorData()
    {
        System.out.println("PDR Vector Data saving started..");

//        List<Document> realDocument = pdfData.LoadPdf();
//        realDocument.forEach(
//                System.out::println
//        );

        // Ordinary chunking: split by a 100-token window and punctuation, without comparing meaning.
        // var splitter = new TokenTextSplitter(100,200,10,5000,true, List.of('.', '?', '!', '\n'));
        // List<Document> tranformedDocuments = splitter.transform(pdfData.LoadPdf());

        // Semantic chunking: embed each sentence and start a new chunk where the topic changes.
        List<Document> tranformedDocuments = semanticChunk(pdfData.LoadPdf());
        for (Document document : tranformedDocuments) {
            document.getMetadata().putAll(embeddingMetadata("sample.pdf"));
        }

        deleteBySource("sample.pdf");
        vectorStore.add(tranformedDocuments);
        storeVectorInMetadata(tranformedDocuments);
        System.out.println("PDF Data saving Completed");
    }

    private List<Document> semanticChunk(List<Document> pages) {
        List<String> sentences = new ArrayList<>();
        List<Map<String, Object>> sentenceMetadata = new ArrayList<>();
        for (Document page : pages) {
            Map<String, Object> pageMetadata = new LinkedHashMap<>(page.getMetadata());
            for (String sentence : splitSentences(page.getText())) {
                sentences.add(sentence);
                sentenceMetadata.add(pageMetadata);
            }
        }
        if (sentences.isEmpty()) {
            return List.of();
        }
        if (sentences.size() == 1) {
            return List.of(chunkDocument(sentences.get(0), sentenceMetadata.get(0), 0));
        }

        List<float[]> embeddings = embeddingClient.embed(sentences);
        double[] distances = new double[sentences.size() - 1];
        for (int i = 0; i < distances.length; i++) {
            distances[i] = 1.0 - cosineSimilarity(embeddings.get(i), embeddings.get(i + 1));
        }
        // Break only at the largest meaning gaps, the 95th percentile of cosine distance.
        double breakpoint = percentile(distances, 95);

        List<Document> chunks = new ArrayList<>();
        StringBuilder currentText = new StringBuilder(sentences.get(0));
        Map<String, Object> currentMetadata = sentenceMetadata.get(0);
        int chunkIndex = 0;
        for (int i = 0; i < distances.length; i++) {
            if (distances[i] > breakpoint) {
                chunks.add(chunkDocument(currentText.toString(), currentMetadata, chunkIndex++));
                currentText = new StringBuilder(sentences.get(i + 1));
                currentMetadata = sentenceMetadata.get(i + 1);
            }
            else {
                currentText.append(' ').append(sentences.get(i + 1));
            }
        }
        chunks.add(chunkDocument(currentText.toString(), currentMetadata, chunkIndex));
        return chunks;
    }

    private Document chunkDocument(String text, Map<String, Object> metadata, int index) {
        Map<String, Object> chunkMetadata = new LinkedHashMap<>(metadata);
        chunkMetadata.put("chunking", "semantic");
        chunkMetadata.put("index", index);
        return new Document(UUID.randomUUID().toString(), text, chunkMetadata);
    }

    private List<String> splitSentences(String text) {
        if (text == null || text.isBlank()) {
            return List.of();
        }
        String normalized = text.replaceAll("\\s+", " ").trim();
        List<String> sentences = new ArrayList<>();
        for (String sentence : normalized.split("(?<=[.!?])\\s+")) {
            if (!sentence.isBlank()) {
                sentences.add(sentence.trim());
            }
        }
        return sentences;
    }

    private double cosineSimilarity(float[] left, float[] right) {
        double dot = 0;
        double leftNorm = 0;
        double rightNorm = 0;
        for (int i = 0; i < left.length; i++) {
            dot += left[i] * (double) right[i];
            leftNorm += left[i] * (double) left[i];
            rightNorm += right[i] * (double) right[i];
        }
        double denominator = Math.sqrt(leftNorm) * Math.sqrt(rightNorm);
        return denominator == 0 ? 0 : dot / denominator;
    }

    private double percentile(double[] values, int percentile) {
        double[] sorted = Arrays.copyOf(values, values.length);
        Arrays.sort(sorted);
        int index = (int) Math.ceil(percentile / 100.0 * sorted.length) - 1;
        index = Math.max(0, Math.min(index, sorted.length - 1));
        return sorted[index];
    }

    private Map<String, Object> embeddingMetadata(String source) {
        Map<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("source", source);
        metadata.put("embeddingModel", embeddingModel);
        metadata.put("dimensions", embeddingDimensions);
        return metadata;
    }

    private void deleteBySource(String source) {
        jdbcTemplate.update(
                "DELETE FROM " + vectorTable() + " WHERE JSON_UNQUOTE(JSON_EXTRACT(metadata, '$.source')) = ?",
                source
        );
    }

    private void storeVectorInMetadata(List<Document> documents) {
        String sql = "UPDATE " + vectorTable()
                + " SET metadata = JSON_SET(metadata, '$.vector', JSON_EXTRACT(VEC_ToText(embedding), '$')) WHERE id = ?";
        jdbcTemplate.batchUpdate(sql, new BatchPreparedStatementSetter() {
            @Override
            public void setValues(PreparedStatement statement, int index) throws SQLException {
                statement.setString(1, documents.get(index).getId());
            }

            @Override
            public int getBatchSize() {
                return documents.size();
            }
        });
    }

    private String vectorTable() {
        if (!vectorTableName.matches("[A-Za-z0-9_]+")) {
            throw new IllegalStateException("Invalid vector table name: " + vectorTableName);
        }
        return vectorTableName;
    }


}
