const express = require('express');
const cors = require('cors');
const { Pinecone } = require('@pinecone-database/pinecone');
const { Pool } = require('pg');
const axios = require('axios');
require('dotenv').config();

// For embeddings - using a lightweight approach
const natural = require('natural');
const TfIdf = natural.TfIdf;

const app = express();
app.use(cors());
app.use(express.json());

// Configuration
const GROQ_API_KEY = process.env.GROQ_API_KEY || '';
const PINECONE_API_KEY = process.env.PINECONE_API_KEY || '';
const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX_NAME || 'everleaf';
const DATABASE_URL = process.env.DATABASE_URL || '';

console.log("=".repeat(50));
console.log("🚀 ENHANCED NODE.JS GROQ SERVER WITH SURGICAL EDITING");
console.log(`🔑 Groq API Key: ${GROQ_API_KEY ? '✅ Ready' : '❌ Missing'}`);
console.log(`🧠 Pinecone API Key: ${PINECONE_API_KEY ? '✅ Ready' : '❌ Missing'}`);
console.log(`💾 Database: ${DATABASE_URL ? '✅ Ready' : '❌ Missing'}`);
console.log("=".repeat(50));

// Initialize services
let pineconeClient = null;
let pineconeIndex = null;
let dbPool = null;

// Initialize Pinecone
async function initPinecone() {
    if (PINECONE_API_KEY && !pineconeClient) {
        try {
            pineconeClient = new Pinecone({ 
                apiKey: PINECONE_API_KEY
            });
            pineconeIndex = pineconeClient.index(PINECONE_INDEX_NAME);
            console.log("✅ Pinecone initialized successfully");
        } catch (error) {
            console.error("❌ Pinecone initialization failed:", error.message);
            try {
                pineconeClient = new Pinecone({ apiKey: PINECONE_API_KEY });
                pineconeIndex = pineconeClient.index(PINECONE_INDEX_NAME);
                console.log("✅ Pinecone initialized successfully (v2 format)");
            } catch (error2) {
                console.error("❌ Pinecone v2 initialization also failed:", error2.message);
            }
        }
    }
    return pineconeIndex;
}

// Initialize Database
function initDatabase() {
    if (DATABASE_URL && !dbPool) {
        try {
            dbPool = new Pool({
                connectionString: DATABASE_URL,
                ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false
            });
            console.log("✅ Database pool initialized");
        } catch (error) {
            console.error("❌ Database initialization failed:", error.message);
        }
    }
    return dbPool;
}

// Simple embedding function using TF-IDF and mathematical transformation
function createSimpleEmbedding(text) {
    console.log(`🎯 Creating simple embedding for: ${text.substring(0, 50)}...`);
    
    // Normalize text
    const normalizedText = text.toLowerCase().replace(/[^\w\s]/g, ' ').trim();
    const words = normalizedText.split(/\s+/).filter(word => word.length > 2);
    
    // Create 1024-dimensional embedding
    const embedding = new Array(1024).fill(0.0);
    
    // Use words to influence embedding values
    words.slice(0, 100).forEach((word, i) => {
        [...word.slice(0, 10)].forEach((char, j) => {
            const charCode = char.charCodeAt(0);
            const embeddingIndex = (charCode + i * 7 + j * 3) % 1024;
            embedding[embeddingIndex] += charCode / 1000.0;
        });
    });
    
    // Normalize the embedding
    const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    if (magnitude > 0) {
        for (let i = 0; i < embedding.length; i++) {
            embedding[i] /= magnitude;
        }
    }
    
    console.log(`✅ Simple embedding created: ${embedding.length} dimensions`);
    return embedding;
}

// Enhanced embedding using TF-IDF approach
function generateEnhancedEmbedding(text) {
    console.log(`🧠 Generating enhanced embedding for: ${text.substring(0, 50)}...`);
    
    try {
        // Create TF-IDF vector
        const tfidf = new TfIdf();
        tfidf.addDocument(text);
        
        // Get term frequencies
        const terms = {};
        tfidf.listTerms(0).forEach((item) => {
            terms[item.term] = item.tfidf;
        });
        
        // Create base embedding from simple method
        const baseEmbedding = createSimpleEmbedding(text);
        
        // Enhance with TF-IDF weights
        const words = text.toLowerCase().split(/\s+/);
        words.forEach((word, i) => {
            if (terms[word] && i < 100) {
                const weight = terms[word];
                const index = (word.charCodeAt(0) + i * 13) % 1024;
                baseEmbedding[index] = Math.min(1.0, baseEmbedding[index] + weight * 0.1);
            }
        });
        
        console.log("✅ Enhanced embedding generated successfully");
        return baseEmbedding;
        
    } catch (error) {
        console.error("❌ Enhanced embedding failed, using simple embedding:", error.message);
        return createSimpleEmbedding(text);
    }
}

// NEW: Detect if prompt is surgical editing request
function detectSurgicalEditingRequest(message) {
    const surgicalKeywords = [
        'expand', 'delete', 'remove', 'replace', 'fix', 'improve', 'add to',
        'section', 'subsection', 'introduction', 'conclusion', 'methodology',
        'modify', 'update', 'change', 'rewrite', 'enhance'
    ];
    
    const messageLower = message.toLowerCase();
    const hasSurgicalKeywords = surgicalKeywords.some(keyword => messageLower.includes(keyword));
    
    // Detect section references
    const hasSectionReference = /section\s+\d+|intro|conclusion|methodology|results|literature|background/.test(messageLower);
    
    // Detect targeted editing language
    const hasTargetedLanguage = /\b(this|that|the\s+\w+\s+section|selected|highlighted)\b/.test(messageLower);
    
    return {
        isSurgical: hasSurgicalKeywords || hasSectionReference,
        hasTarget: hasSectionReference || hasTargetedLanguage,
        confidence: (hasSurgicalKeywords ? 0.4 : 0) + (hasSectionReference ? 0.4 : 0) + (hasTargetedLanguage ? 0.2 : 0)
    };
}

// ENHANCED: Create surgical editing system prompt
function createSurgicalSystemPrompt(surgicalAnalysis, modelType) {
    let basePrompt = '';
    
    if (surgicalAnalysis.isSurgical && surgicalAnalysis.confidence > 0.5) {
        basePrompt = `You are a SURGICAL LaTeX editor. Your job is to make PRECISE, TARGETED changes to LaTeX documents.

CRITICAL SURGICAL EDITING RULES:
1. Make ONLY the requested changes - nothing more, nothing less
2. If asked to modify a section, return ONLY that section's content
3. If asked to add content, return ONLY the new content to be added
4. If asked to delete, return empty string or confirmation comment
5. Preserve ALL existing LaTeX syntax and formatting
6. DO NOT add \\documentclass, \\begin{document}, or \\end{document} unless specifically requested
7. DO NOT include content that already exists in the document
8. Focus on the specific target mentioned in the request

RESPONSE FORMAT EXAMPLES:
- For section replacement: Return complete section with \\section{Title} header
- For content addition: Return only the new content to be inserted
- For deletions: Return "% Content deleted" or empty string
- For improvements: Return the improved version of the specified content only

VALIDATION:
- Always maintain LaTeX syntax correctness
- Preserve document structure
- Keep changes minimal and targeted

Your response will be surgically inserted into the document, so be precise.`;
    } else {
        basePrompt = `You are a helpful LaTeX assistant. Generate clean, proper LaTeX code that can be directly used in documents.

IMPORTANT:
- Provide working LaTeX code
- DO NOT include \\documentclass, \\begin{document}, or \\end{document} unless specifically requested
- Use proper LaTeX syntax and formatting
- Make code ready for direct insertion into existing documents`;
    }
    
    // Add model-specific instructions
    if (modelType === 'code') {
        basePrompt += `\n\nFocus on generating clean, syntactically correct LaTeX code with proper formatting.`;
    }
    
    return basePrompt;
}

// ENHANCED: Context formatting for surgical editing
function formatSurgicalContext(contextChunks, originalMessage, surgicalAnalysis) {
    if (!contextChunks || contextChunks.length === 0) {
        return "";
    }
    
    let contextParts = [];
    
    if (surgicalAnalysis.isSurgical) {
        contextParts.push("=== DOCUMENT CONTEXT FOR SURGICAL EDITING ===");
        contextParts.push("Use this context to understand the existing document structure and content.");
        contextParts.push("Make precise changes based on the user's request.\n");
    } else {
        contextParts.push("=== RELEVANT DOCUMENT INFORMATION ===\n");
    }
    
    contextChunks.slice(0, 5).forEach((chunk, i) => {
        const docName = chunk.document?.filename || 'Document';
        contextParts.push(`[SOURCE ${i + 1} - ${docName}]`);
        contextParts.push(chunk.text);
        contextParts.push("---");
    });
    
    if (surgicalAnalysis.isSurgical) {
        contextParts.push("=== END CONTEXT - APPLY SURGICAL CHANGES ONLY ===\n");
    } else {
        contextParts.push("=== END DOCUMENT INFORMATION ===\n");
    }
    
    return contextParts.join('\n');
}

// Query embeddings from Pinecone
async function queryEmbeddings(queryText, namespace, projectId, topK = 5, threshold = -0.8) {
    try {
        const index = await initPinecone();
        if (!index) {
            console.log("❌ Pinecone index not available");
            return [];
        }
        
        console.log(`🔍 Querying embeddings for: ${queryText.substring(0, 50)}...`);
        console.log(`🎯 Namespace: ${namespace}`);
        console.log(`🎯 Threshold: ${threshold}`);
        
        // Generate query embedding
        const queryEmbedding = generateEnhancedEmbedding(queryText);
        if (!queryEmbedding) {
            console.log("❌ Failed to generate query embedding");
            return [];
        }
        
        console.log(`✅ Query embedding generated: ${queryEmbedding.length} dimensions`);
        
        // Query Pinecone with v2.x format
        const queryRequest = {
            vector: queryEmbedding,
            topK: topK * 2,
            includeMetadata: true
        };
        
        console.log(`🔍 Pinecone query request:`, {
            vectorLength: queryEmbedding.length,
            topK: queryRequest.topK,
            namespace: namespace,
            includeMetadata: queryRequest.includeMetadata
        });
        
        const results = await index.namespace(namespace).query(queryRequest);
        
        console.log(`📊 Raw Pinecone results: ${results.matches?.length || 0} matches`);
        
        // Log all scores for debugging
        if (results.matches && results.matches.length > 0) {
            console.log("🔍 All match scores:");
            results.matches.forEach((match, i) => {
                console.log(`   Match ${i + 1}: Score = ${match.score?.toFixed(4)}, ID = ${match.id}`);
            });
        } else {
            console.log("⚠️  No matches returned from Pinecone");
        }
        
        // Filter by threshold and extract relevant info
        const relevantChunks = [];
        if (results.matches) {
            for (const match of results.matches) {
                const chunkData = {
                    id: match.id,
                    score: match.score,
                    text: match.metadata?.text || '',
                    document_id: match.metadata?.documentId || null,
                    chunk_index: match.metadata?.chunkIndex || 0
                };
                
                if (match.score >= threshold) {
                    relevantChunks.push(chunkData);
                    console.log(`✅ Added chunk: Score=${match.score?.toFixed(4)}, Text preview: ${chunkData.text.substring(0, 50)}...`);
                } else {
                    console.log(`❌ Filtered out: Score=${match.score?.toFixed(4)} < threshold=${threshold}`);
                }
            }
        }
        
        console.log(`✅ Found ${relevantChunks.length} relevant chunks (threshold >= ${threshold})`);
        return relevantChunks.slice(0, topK);
        
    } catch (error) {
        console.error("❌ Embedding query error:", error.message);
        return [];
    }
}

// Get project namespaces from database
async function getProjectNamespaces(projectId) {
    try {
        const pool = initDatabase();
        if (!pool) {
            console.log("❌ No database pool available");
            return [];
        }
        
        const client = await pool.connect();
        try {
            console.log(`🔍 Querying database for project_id: ${projectId}`);
            
            const result = await client.query(`
                SELECT DISTINCT pinecone_namespace
                FROM project_documents
                WHERE project_id = $1 AND processing_status = 'completed'
            `, [projectId]);
            
            const namespaces = result.rows.map(row => row.pinecone_namespace);
            console.log(`📊 Found ${namespaces.length} namespaces for project ${projectId}:`, namespaces);
            return namespaces;
            
        } finally {
            client.release();
        }
        
    } catch (error) {
        console.error("❌ Database query error:", error.message);
        console.error("Full error:", error);
        return [];
    }
}

// Get relevant context from all project documents
async function getRelevantContext(projectId, queryText, maxChunks = 5) {
    try {
        console.log(`🔍 Getting relevant context for project ${projectId}`);
        
        // Get all namespaces for the project
        const namespaces = await getProjectNamespaces(projectId);
        if (namespaces.length === 0) {
            console.log("⚠️ No completed documents found for project");
            return { chunks: [], document_ids: [] };
        }
        
        const allChunks = [];
        
        // Query each namespace
        for (const namespace of namespaces) {
            console.log(`🔍 Searching namespace: ${namespace}`);
            const chunks = await queryEmbeddings(queryText, namespace, projectId, maxChunks * 2);
            allChunks.push(...chunks);
        }
        
        // Sort by score and take top chunks
        allChunks.sort((a, b) => b.score - a.score);
        const topChunks = allChunks.slice(0, maxChunks);
        
        // Get document info
        const documentIds = [...new Set(topChunks.map(chunk => chunk.document_id).filter(Boolean))];
        
        const documentInfo = {};
        if (documentIds.length > 0) {
            const pool = initDatabase();
            if (pool) {
                const client = await pool.connect();
                try {
                    const result = await client.query(`
                        SELECT id, original_filename, upload_date
                        FROM project_documents
                        WHERE id = ANY($1)
                    `, [documentIds]);
                    
                    result.rows.forEach(doc => {
                        documentInfo[doc.id] = {
                            filename: doc.original_filename,
                            upload_date: doc.upload_date
                        };
                    });
                } finally {
                    client.release();
                }
            }
        }
        
        // Add document info to chunks
        topChunks.forEach(chunk => {
            if (chunk.document_id in documentInfo) {
                chunk.document = documentInfo[chunk.document_id];
            }
        });
        
        const result = {
            chunks: topChunks,
            document_ids: documentIds
        };
        
        console.log(`✅ Context retrieval completed: ${topChunks.length} chunks`);
        return result;
        
    } catch (error) {
        console.error("❌ Context retrieval error:", error.message);
        return { chunks: [], document_ids: [] };
    }
}

// NEW: Post-process surgical responses
function postProcessSurgicalResponse(response, originalMessage, surgicalAnalysis) {
    // Remove common prefixes that aren't needed
    let cleanResponse = response.trim();
    
    // Remove explanatory text that might interfere with surgical editing
    const explanatoryPhrases = [
        'Here is the',
        'Here\'s the',
        'I\'ve',
        'I have',
        'This is the',
        'The following is',
        'Below is the'
    ];
    
    for (const phrase of explanatoryPhrases) {
        if (cleanResponse.toLowerCase().startsWith(phrase.toLowerCase())) {
            // Find the first newline or colon after the phrase
            const breakIndex = Math.min(
                cleanResponse.indexOf('\n') > -1 ? cleanResponse.indexOf('\n') : Infinity,
                cleanResponse.indexOf(':') > -1 ? cleanResponse.indexOf(':') : Infinity
            );
            if (breakIndex < Infinity) {
                cleanResponse = cleanResponse.substring(breakIndex + 1).trim();
            }
        }
    }
    
    // For deletion requests, ensure clean response
    if (originalMessage.toLowerCase().includes('delete') || originalMessage.toLowerCase().includes('remove')) {
        if (cleanResponse.toLowerCase().includes('deleted') || cleanResponse.toLowerCase().includes('removed')) {
            return '% Content deleted';
        }
    }
    
    return cleanResponse;
}

// ENHANCED: Groq request with better temperature control
async function makeEnhancedGroqRequest(message, context, modelType, surgicalAnalysis) {
    if (!GROQ_API_KEY) {
        return "❌ Get free Groq API key at: https://console.groq.com/";
    }
    
    // Create surgical system prompt
    const systemPrompt = createSurgicalSystemPrompt(surgicalAnalysis, modelType);
    
    // Format message with context
    let fullMessage = message;
    if (context) {
        fullMessage = `${context}\n\nUSER REQUEST: ${message}`;
        
        if (surgicalAnalysis.isSurgical) {
            fullMessage += `\n\nIMPORTANT: This is a surgical editing request. Provide only the specific content needed for the requested change.`;
        }
    }
    
    // Adjust parameters based on surgical nature
    const temperature = surgicalAnalysis.isSurgical ? 0.05 : 0.3; // Much lower for surgical edits
    const maxTokens = surgicalAnalysis.isSurgical ? 1500 : 2000;
    
    const models = {
        "code": "llama-3.1-8b-instant",
        "text": "llama-3.1-8b-instant", 
        "chat": "llama-3.1-8b-instant"
    };
    
    const model = models[modelType] || "llama-3.1-8b-instant";
    
    try {
        console.log(`🤖 Making Enhanced Groq request:`);
        console.log(`   - Model: ${model}`);
        console.log(`   - Temperature: ${temperature}`);
        console.log(`   - Max Tokens: ${maxTokens}`);
        console.log(`   - Surgical: ${surgicalAnalysis.isSurgical}`);
        console.log(`   - Confidence: ${surgicalAnalysis.confidence}`);
        
        const response = await axios.post(
            "https://api.groq.com/openai/v1/chat/completions",
            {
                model,
                messages: [
                    { role: "system", content: systemPrompt },
                    { role: "user", content: fullMessage }
                ],
                max_tokens: maxTokens,
                temperature,
                // Enhanced parameters for surgical editing
                top_p: surgicalAnalysis.isSurgical ? 0.1 : 0.9,
                frequency_penalty: surgicalAnalysis.isSurgical ? 0.2 : 0.0,
                presence_penalty: surgicalAnalysis.isSurgical ? 0.1 : 0.0
            },
            {
                headers: {
                    "Authorization": `Bearer ${GROQ_API_KEY}`,
                    "Content-Type": "application/json"
                },
                timeout: 45000
            }
        );
        
        if (response.data?.choices?.[0]?.message?.content) {
            const aiResponse = response.data.choices[0].message.content.trim();
            console.log(`✅ Enhanced Groq response received (${aiResponse.length} chars)`);
            
            // Post-process response for surgical editing
            if (surgicalAnalysis.isSurgical) {
                console.log('🔧 Post-processing surgical response...');
                return postProcessSurgicalResponse(aiResponse, message, surgicalAnalysis);
            }
            
            return aiResponse;
        }
        
        return `❌ Groq Error: No response content`;
        
    } catch (error) {
        console.error(`❌ Enhanced Groq API error:`, error.response?.data || error.message);
        return `❌ Error: ${error.message}`;
    }
}

// NEW: Enhanced ask Groq with surgical editing support
async function askGroqWithSurgicalEditing(message, contextChunks, modelType = "text") {
    // Analyze if this is a surgical editing request
    const surgicalAnalysis = detectSurgicalEditingRequest(message);
    
    console.log(`🔍 Surgical analysis:`, surgicalAnalysis);
    
    // Format context with surgical awareness
    const formattedContext = formatSurgicalContext(contextChunks, message, surgicalAnalysis);
    
    // Make enhanced request
    return await makeEnhancedGroqRequest(message, formattedContext, modelType, surgicalAnalysis);
}

// Routes
app.get('/health', (req, res) => {
    res.json({
        status: "healthy",
        groq_configured: !!GROQ_API_KEY,
        pinecone_configured: !!PINECONE_API_KEY,
        db_configured: !!DATABASE_URL,
        surgical_editing: true, // NEW: Indicate surgical editing support
        timestamp: new Date().toISOString()
    });
});

app.get('/test', async (req, res) => {
    try {
        // Test surgical editing detection
        const testMessage = "expand the introduction section with more background";
        const surgicalAnalysis = detectSurgicalEditingRequest(testMessage);
        
        // Test Groq
        const groqResult = await askGroqWithSurgicalEditing("Write a simple hello world function in Python", [], "code");
        
        // Test database
        const pool = initDatabase();
        const dbWorking = !!pool;
        
        // Test Pinecone
        const pineconeWorking = !!(await initPinecone());
        
        res.json({
            groq_test: groqResult,
            groq_working: !groqResult.startsWith("❌"),
            database_working: dbWorking,
            pinecone_working: pineconeWorking,
            surgical_test: surgicalAnalysis, // NEW: Show surgical detection test
            timestamp: new Date().toISOString()
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// ENHANCED: Main chat endpoint with surgical editing
app.post('/chat', async (req, res) => {
    try {
        const { message, model_type = 'text', project_id, use_rag = false } = req.body;
        
        if (!message?.trim()) {
            return res.status(400).json({ error: "Message required" });
        }
        
        console.log(`💬 Enhanced Chat Request:`);
        console.log(`   - Message: ${message.substring(0, 100)}...`);
        console.log(`   - Project ID: ${project_id}`);
        console.log(`   - Use RAG: ${use_rag}`);
        console.log(`   - Model Type: ${model_type}`);
        
        let contextChunks = [];
        
        // Get relevant context if RAG is enabled
        if (use_rag && project_id) {
            console.log(`🔍 Getting context for project ${project_id}`);
            const contextResult = await getRelevantContext(project_id, message);
            contextChunks = contextResult.chunks || [];
            console.log(`📊 Found ${contextChunks.length} relevant chunks`);
        }
        
        // Use enhanced surgical editing approach
        const response = await askGroqWithSurgicalEditing(message, contextChunks, model_type);
        
        // Analyze the request for metadata
        const surgicalAnalysis = detectSurgicalEditingRequest(message);
        
        console.log(`✅ Enhanced response generated`);
        
        res.json({
            response,
            model_type,
            context_used: contextChunks.length,
            success: true,
            surgical_analysis: surgicalAnalysis, // NEW: Include surgical analysis
            metadata: {
                is_surgical: surgicalAnalysis.isSurgical,
                confidence: surgicalAnalysis.confidence,
                has_target: surgicalAnalysis.hasTarget
            },
            timestamp: new Date().toISOString()
        });
        
    } catch (error) {
        console.error('❌ Enhanced chat error:', error);
        res.status(500).json({ error: error.message });
    }
});

// ENHANCED: LaTeX assist with surgical editing
app.post('/latex-assist', async (req, res) => {
    try {
        const { action, selected_text = '', request: userRequest = '', project_id, use_rag = true } = req.body;
        
        if (!action) {
            return res.status(400).json({ error: "Action required" });
        }
        
        // Create surgical prompts based on action
        let prompt;
        let isSurgicalAction = true;
        
        switch (action) {
            case 'generate':
                prompt = `Generate LaTeX code for: ${userRequest}`;
                isSurgicalAction = false;
                break;
            case 'fix':
                prompt = `Fix this LaTeX code surgically - return only the corrected version:\n\n${selected_text}\n\nIssue to fix: ${userRequest}`;
                break;
            case 'explain':
                prompt = `Explain this LaTeX code concisely:\n\n${selected_text}`;
                isSurgicalAction = false;
                break;
            case 'improve':
                prompt = `Improve this LaTeX code surgically - return only the improved version:\n\n${selected_text}\n\nImprovement focus: ${userRequest}`;
                break;
            case 'complete':
                prompt = `Complete this LaTeX code - return only the additional content needed:\n\n${selected_text}\n\nCompletion request: ${userRequest}`;
                break;
            case 'delete':
                prompt = `Confirm deletion of this LaTeX code:\n\n${selected_text}\n\nReturn "% Content deleted" to confirm or explain why it shouldn't be deleted.`;
                break;
            case 'expand':
                prompt = `Expand this LaTeX content - return only the new content to add:\n\n${selected_text}\n\nExpansion request: ${userRequest}`;
                break;
            default:
                return res.status(400).json({ error: "Invalid action" });
        }
        
        console.log(`🔧 Enhanced LaTeX action: ${action}`);
        console.log(`🎯 Project ID: ${project_id}`);
        console.log(`🧠 Use RAG: ${use_rag}`);
        console.log(`🔪 Surgical: ${isSurgicalAction}`);
        
        let contextChunks = [];
        
        // Get relevant context if RAG is enabled
        if (use_rag && project_id) {
            console.log(`🔍 Getting LaTeX context for project ${project_id}`);
            const contextResult = await getRelevantContext(project_id, prompt);
            contextChunks = contextResult.chunks || [];
            console.log(`📊 Found ${contextChunks.length} relevant chunks for LaTeX`);
        }
        
        // Create surgical analysis for this action
        const surgicalAnalysis = {
            isSurgical: isSurgicalAction,
            confidence: isSurgicalAction ? 0.9 : 0.1,
            hasTarget: !!selected_text,
            action: action
        };
        
        // Use enhanced request for LaTeX
        const response = await makeEnhancedGroqRequest(prompt, 
            formatSurgicalContext(contextChunks, prompt, surgicalAnalysis), 
            "code", 
            surgicalAnalysis
        );
        
        res.json({
            response,
            action,
            context_used: contextChunks.length,
            success: true,
            surgical: isSurgicalAction, // Indicate if this was surgical
            metadata: {
                selected_text_length: selected_text.length,
                has_context: contextChunks.length > 0,
                surgical_analysis: surgicalAnalysis
            },
            timestamp: new Date().toISOString()
        });
        
    } catch (error) {
        console.error('❌ Enhanced LaTeX assist error:', error);
        res.status(500).json({ error: error.message });
    }
});

// Context query endpoints (unchanged)
app.post('/api/chat/projects/:projectId/query-context', async (req, res) => {
    try {
        const { projectId } = req.params;
        const { query, maxChunks = 5 } = req.body;
        
        if (!query?.trim()) {
            return res.status(400).json({ error: "Query required" });
        }
        
        console.log(`🔍 Querying context: ${query.substring(0, 50)}...`);
        console.log(`🎯 Project ID: ${projectId}`);
        
        const contextResult = await getRelevantContext(parseInt(projectId), query, maxChunks);
        
        res.json({
            success: true,
            context: contextResult,
            message: `Found ${contextResult.chunks?.length || 0} relevant chunks`
        });
        
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.post('/query-context', async (req, res) => {
    try {
        const { query, project_id, max_chunks = 5 } = req.body;
        
        if (!query?.trim() || !project_id) {
            return res.status(400).json({ error: "Query and project_id required" });
        }
        
        console.log(`🔍 Querying context: ${query.substring(0, 50)}...`);
        console.log(`🎯 Project ID: ${project_id}`);
        
        const contextResult = await getRelevantContext(project_id, query, max_chunks);
        
        res.json({
            context: contextResult,
            success: true,
            timestamp: new Date().toISOString()
        });
        
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Initialize services and start server
async function startServer() {
    const port = process.env.PORT || 5001;
    
    // Initialize services
    await initPinecone();
    initDatabase();
    
    // Check for missing configurations
    const missingConfigs = [];
    if (!GROQ_API_KEY) missingConfigs.push("GROQ_API_KEY");
    if (!PINECONE_API_KEY) missingConfigs.push("PINECONE_API_KEY");
    if (!DATABASE_URL) missingConfigs.push("DATABASE_URL");
    
    if (missingConfigs.length > 0) {
        console.log("\n⚠️  MISSING CONFIGURATION:");
        missingConfigs.forEach(config => {
            console.log(`📝 Add to .env: ${config}=your_key_here`);
        });
        console.log("🔄 Then restart server\n");
    }
    
    app.listen(port, '0.0.0.0', () => {
        console.log(`🚀 Enhanced Node.js server with SURGICAL EDITING running on port ${port}`);
        console.log(`🔪 Features: RAG, Surgical LaTeX Editing, Intent Detection`);
    });
}

// Handle graceful shutdown
process.on('SIGTERM', () => {
    console.log('🛑 Received SIGTERM, shutting down gracefully');
    if (dbPool) {
        dbPool.end();
    }
    process.exit(0);
});

startServer().catch(console.error);