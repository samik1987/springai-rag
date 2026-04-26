package com.example.springai_rag.configs;

import com.example.springai_rag.services.VectorStoreService;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor;
import org.springframework.ai.chat.client.advisor.SafeGuardAdvisor;
import org.springframework.ai.chat.client.advisor.SimpleLoggerAdvisor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.memory.InMemoryChatMemoryRepository;
import org.springframework.ai.chat.memory.MessageWindowChatMemory;
import org.springframework.ai.openai.OpenAiChatModel;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.List;

@Configuration
public class VectorConfig {

    @Bean
    public VectorStoreService getVectorService()
    {
        return new VectorStoreService();
    }

    @Bean
    public ChatMemory getInMemoryChatMemory()
    {
        InMemoryChatMemoryRepository inMemoryRepo = new InMemoryChatMemoryRepository();
        return MessageWindowChatMemory.builder().maxMessages(10).chatMemoryRepository(inMemoryRepo).build();
    }

    @Bean(name= "OpenAiChatClient")
    public ChatClient openAiClient(OpenAiChatModel model , ChatMemory chatMemory) {

        MessageChatMemoryAdvisor messageChatMemoryAdvisor = MessageChatMemoryAdvisor.builder(chatMemory).build();

        return ChatClient.builder(model).defaultAdvisors(
                messageChatMemoryAdvisor,
                new SimpleLoggerAdvisor(),
                new SafeGuardAdvisor(List.of("sex" ,"porn"))
        ).build();
    }

    @Bean(name= "QuesAnsChatClient")
    public ChatClient QuesAnsChatClient(OpenAiChatModel model , ChatMemory chatMemory) {

        MessageChatMemoryAdvisor messageChatMemoryAdvisor = MessageChatMemoryAdvisor.builder(chatMemory).build();

        return ChatClient.builder(model).defaultAdvisors(
                messageChatMemoryAdvisor,
                new SimpleLoggerAdvisor(),
                new SafeGuardAdvisor(List.of("sex" ,"porn"))
        ).build();
    }
}
