package com.example.springai_rag.services;

import com.example.springai_rag.db.ListData;
import com.example.springai_rag.db.PdfData;
import org.springframework.ai.document.Document;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Autowired;

import java.util.List;

public class VectorStoreService {

    @Autowired
    private VectorStore vectorStore;

    @Autowired
    private PdfData pdfData;

    public void SaveVectorData()
    {
        System.out.println("Vector Data saving started..");
        List<Document> documents = ListData.getData().stream().map(Document::new).toList();
        vectorStore.add(documents);
        System.out.println("Data saving Completed");
    }

    public  List<Document> doSimilaritySearch(String msg)
    {
        SearchRequest srchReq = SearchRequest.builder().topK(5)
                .similarityThreshold(0.6)
                .query(msg)
                .build();

        return vectorStore.similaritySearch(srchReq);
    }

    public  void SavePdfVectorData()
    {
        System.out.println("PDR Vector Data saving started..");

        List<Document> realDocument = pdfData.LoadPdf();
        realDocument.forEach(
                System.out::println
        );

        var splitter = new TokenTextSplitter(100,200,10,5000,true, List.of('.', '?', '!', '\n'));

        List<Document> tranformedDocuments = splitter.transform(pdfData.LoadPdf());

        vectorStore.add(tranformedDocuments);
        System.out.println("PDF Data saving Completed");
    }


}
