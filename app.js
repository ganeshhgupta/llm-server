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
console.log("🚀 NODE.JS GROQ SERVER WITH RAG");
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
            pineconeClient = new Pinecone({ apiKey: PINECONE_API_KEY });
            pineconeIndex = pineconeClient.index(PINECONE_INDEX_NAME);
            console.log("✅ Pinecone initialized successfully");
        } catch (error) {
            console.error("❌ Pinecone initialization failed:", error.message);
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
        
        // Query Pinecone with correct format
        const queryRequest = {
            vector: queryEmbedding,
            topK: topK * 2,
            includeMetadata: true,
            namespace: namespace
        };
        
        console.log(`🔍 Pinecone query request:`, {
            vectorLength: queryEmbedding.length,
            topK: queryRequest.topK,
            namespace: queryRequest.namespace,
            includeMetadata: queryRequest.includeMetadata
        });
        
        const results = await index.query(queryRequest);
        
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
            
            // First, let's see what's actually in the table
            const allDocs = await client.query(`
                SELECT project_id, pinecone_namespace, processing_status, original_filename
                FROM project_documents
                WHERE project_id = $1
                LIMIT 10
            `, [projectId]);
            
            console.log(`📋 All documents for project ${projectId}:`, allDocs.rows);
            
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

// Analyze query intent
function analyzeQueryIntent(message) {
    const messageLower = message.toLowerCase();
    
    const analysis = {
        query_type: 'general',
        requires_multi_step: false,
        info_needed: [],
        temporal_aspect: 'current',
        specificity: 'specific',
        complexity: 'simple'
    };
    
    // Detect query type
    if (['where', 'which', 'what university', 'what college', 'school'].some(word => messageLower.includes(word))) {
        analysis.query_type = 'location_education';
        analysis.info_needed = ['university', 'college', 'school', 'education', 'degree'];
    } else if (['what degree', 'studying', 'pursuing', 'major'].some(word => messageLower.includes(word))) {
        analysis.query_type = 'degree_program';
        analysis.info_needed = ['degree', 'major', 'program', 'thesis', 'specialization'];
    } else if (['experience', 'work', 'job', 'position', 'role'].some(word => messageLower.includes(word))) {
        analysis.query_type = 'professional_experience';
        analysis.info_needed = ['work', 'job', 'position', 'experience', 'company'];
    } else if (['skills', 'technology', 'programming', 'tools'].some(word => messageLower.includes(word))) {
        analysis.query_type = 'technical_skills';
        analysis.info_needed = ['skills', 'technology', 'programming', 'tools', 'languages'];
    }
    
    // Detect temporal aspect
    if (['graduated', 'completed', 'finished', 'past'].some(word => messageLower.includes(word))) {
        analysis.temporal_aspect = 'past';
    } else if (['currently', 'now', 'present', 'studying', 'pursuing'].some(word => messageLower.includes(word))) {
        analysis.temporal_aspect = 'current';
    } else if (['will', 'future', 'plan', 'next'].some(word => messageLower.includes(word))) {
        analysis.temporal_aspect = 'future';
    }
    
    // Detect complexity
    if (messageLower.includes(' and ') || message.split(' ').length > 10) {
        analysis.complexity = 'complex';
        analysis.requires_multi_step = true;
    }
    
    return analysis;
}

// Format intelligent context
function formatIntelligentContext(contextChunks, queryAnalysis, originalMessage) {
    if (!contextChunks || contextChunks.length === 0) {
        return "";
    }
    
    // Score chunks based on relevance
    const scoredChunks = contextChunks.map(chunk => ({
        chunk,
        score: calculateChunkRelevanceScore(chunk, queryAnalysis, originalMessage)
    }));
    
    // Sort by relevance score
    scoredChunks.sort((a, b) => b.score - a.score);
    
    // Format context
    const contextParts = [];
    contextParts.push("=== RELEVANT INFORMATION FROM DOCUMENTS ===\n");
    
    scoredChunks.slice(0, 5).forEach((item, i) => {
        const { chunk, score } = item;
        const docName = chunk.document?.filename || 'Unknown document';
        
        contextParts.push(`[SOURCE ${i + 1} - ${docName} - Relevance: ${score.toFixed(2)}]`);
        contextParts.push(chunk.text);
        contextParts.push("---");
    });
    
    contextParts.push("=== END OF DOCUMENT INFORMATION ===\n");
    
    return contextParts.join('\n');
}

// Calculate chunk relevance score
function calculateChunkRelevanceScore(chunk, queryAnalysis, originalMessage) {
    const chunkText = chunk.text.toLowerCase();
    let score = 0.0;
    
    // Base score from similarity
    const baseScore = chunk.score || 0.0;
    score += baseScore * 10;
    
    // Keyword matching based on query type
    queryAnalysis.info_needed.forEach(keyword => {
        if (chunkText.includes(keyword)) {
            score += 5.0;
        }
    });
    
    // Specific term matching
    const queryWords = originalMessage.toLowerCase().split(' ');
    queryWords.forEach(word => {
        if (word.length > 3 && chunkText.includes(word)) {
            score += 2.0;
        }
    });
    
    // Bonus for certain key phrases based on query type
    if (queryAnalysis.query_type === 'location_education') {
        if (['university', 'college', 'teaching assistant', 'graduate'].some(phrase => chunkText.includes(phrase))) {
            score += 10.0;
        }
    } else if (queryAnalysis.query_type === 'degree_program') {
        if (['master', 'degree', 'thesis', 'specialization'].some(phrase => chunkText.includes(phrase))) {
            score += 10.0;
        }
    }
    
    return score;
}

// Create adaptive system prompt
function createAdaptiveSystemPrompt(queryAnalysis, modelType) {
    const basePrompt = `You are an intelligent document analysis assistant. Your job is to carefully analyze the provided document information and answer questions accurately based on ONLY the information present in the documents.

CRITICAL INSTRUCTIONS:
1. Read ALL the provided document sources carefully
2. Look for information that directly answers the user's question
3. If you find relevant information, state it clearly and mention which document it came from
4. If information is incomplete, say what you found and what's missing
5. NEVER say "I don't have information" if relevant details are actually present in the sources
6. Pay special attention to employment, education, and personal details`;
    
    let specificPrompt = "";
    
    switch (queryAnalysis.query_type) {
        case 'location_education':
            specificPrompt = `
SPECIFIC FOCUS: The user is asking about educational institutions, universities, colleges, or schools.
- Look for terms like: university, college, school, graduate, teaching assistant, student
- Pay attention to institutional names, locations, and current/past affiliations
- Note any degree programs, enrollment status, or educational roles`;
            break;
        case 'degree_program':
            specificPrompt = `
SPECIFIC FOCUS: The user is asking about academic degrees, programs, or specializations.
- Look for terms like: degree, major, thesis, specialization, program, masters, bachelor
- Pay attention to field of study, research topics, and academic focus areas
- Note any current vs. completed educational programs`;
            break;
        case 'professional_experience':
            specificPrompt = `
SPECIFIC FOCUS: The user is asking about work experience, jobs, or professional roles.
- Look for terms like: job, work, position, role, company, employer, experience
- Pay attention to job titles, company names, employment dates, and responsibilities`;
            break;
        default:
            specificPrompt = `
SPECIFIC FOCUS: Provide accurate information based on the document contents.
- Read carefully and extract relevant details
- Be specific and cite sources when possible`;
    }
    
    return basePrompt + specificPrompt;
}

// Make Groq API request
async function makeGroqRequest(message, context, systemPrompt, modelType, queryAnalysis) {
    // Combine message and context
    const fullMessage = context 
        ? `${context}\n\nUSER QUESTION: ${message}\n\nPlease answer based on the document information provided above.`
        : message;
    
    // Adjust temperature based on query type
    const temperature = ['location_education', 'degree_program'].includes(queryAnalysis.query_type) ? 0.1 : 0.3;
    
    const models = {
        "code": "llama-3.1-8b-instant",
        "text": "llama-3.1-8b-instant",
        "chat": "llama-3.1-8b-instant"
    };
    
    const model = models[modelType] || "llama-3.1-8b-instant";
    
    try {
        const response = await axios.post(
            "https://api.groq.com/openai/v1/chat/completions",
            {
                model,
                messages: [
                    { role: "system", content: systemPrompt },
                    { role: "user", content: fullMessage }
                ],
                max_tokens: 2000,
                temperature
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
            return response.data.choices[0].message.content.trim();
        }
        
        return `❌ Groq Error: No response content`;
        
    } catch (error) {
        return `❌ Error: ${error.message}`;
    }
}

// Main function to ask Groq with context
async function askGroqWithContext(message, contextChunks, modelType = "text") {
    if (!GROQ_API_KEY) {
        return "❌ Get free Groq API key at: https://console.groq.com/";
    }
    
    // Analyze query
    const queryAnalysis = analyzeQueryIntent(message);
    
    // Format context
    const formattedContext = formatIntelligentContext(contextChunks, queryAnalysis, message);
    
    // Create system prompt
    const systemPrompt = createAdaptiveSystemPrompt(queryAnalysis, modelType);
    
    // Make request
    return await makeGroqRequest(message, formattedContext, systemPrompt, modelType, queryAnalysis);
}

// Routes
app.get('/health', (req, res) => {
    res.json({
        status: "healthy",
        groq_configured: !!GROQ_API_KEY,
        pinecone_configured: !!PINECONE_API_KEY,
        db_configured: !!DATABASE_URL,
        timestamp: new Date().toISOString()
    });
});

app.get('/test', async (req, res) => {
    try {
        // Test Groq
        const groqResult = await askGroqWithContext("Write a simple hello world function in Python", [], "code");
        
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
            timestamp: new Date().toISOString()
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.post('/chat', async (req, res) => {
    try {
        const { message, model_type = 'text', project_id, use_rag = false } = req.body;
        
        if (!message?.trim()) {
            return res.status(400).json({ error: "Message required" });
        }
        
        console.log(`💬 Question: ${message.substring(0, 50)}...`);
        console.log(`🎯 Project ID: ${project_id}`);
        console.log(`🧠 Use RAG: ${use_rag}`);
        
        let contextChunks = [];
        
        // Get relevant context if RAG is enabled
        if (use_rag && project_id) {
            console.log(`🔍 Getting context for project ${project_id}`);
            const contextResult = await getRelevantContext(project_id, message);
            contextChunks = contextResult.chunks || [];
            console.log(`📊 Found ${contextChunks.length} relevant chunks`);
        }
        
        // Ask Groq with context
        const response = await askGroqWithContext(message, contextChunks, model_type);
        
        console.log(`✅ Answer: ${response.substring(0, 50)}...`);
        
        res.json({
            response,
            model_type,
            context_used: contextChunks.length,
            success: true,
            timestamp: new Date().toISOString()
        });
        
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

app.post('/latex-assist', async (req, res) => {
    try {
        const { action, selected_text = '', request: userRequest = '', project_id, use_rag = true } = req.body;
        
        if (!action) {
            return res.status(400).json({ error: "Action required" });
        }
        
        // Create prompts based on action
        let prompt;
        switch (action) {
            case 'generate':
                prompt = `Generate LaTeX code for: ${userRequest}`;
                break;
            case 'fix':
                prompt = `Fix this LaTeX code:\n${selected_text}\n\nRequest: ${userRequest}`;
                break;
            case 'explain':
                prompt = `Explain this LaTeX code:\n${selected_text}`;
                break;
            case 'improve':
                prompt = `Improve this LaTeX code:\n${selected_text}\n\nRequest: ${userRequest}`;
                break;
            case 'complete':
                prompt = `Complete this LaTeX code:\n${selected_text}\n\nRequest: ${userRequest}`;
                break;
            default:
                return res.status(400).json({ error: "Invalid action" });
        }
        
        console.log(`🔧 LaTeX action: ${action}`);
        console.log(`🎯 Project ID: ${project_id}`);
        console.log(`🧠 Use RAG: ${use_rag}`);
        
        let contextChunks = [];
        
        // Get relevant context if RAG is enabled
        if (use_rag && project_id) {
            console.log(`🔍 Getting LaTeX context for project ${project_id}`);
            const contextResult = await getRelevantContext(project_id, prompt);
            contextChunks = contextResult.chunks || [];
            console.log(`📊 Found ${contextChunks.length} relevant chunks for LaTeX`);
        }
        
        // Ask Groq with context
        const response = await askGroqWithContext(prompt, contextChunks, "code");
        
        res.json({
            response,
            action,
            context_used: contextChunks.length,
            success: true,
            timestamp: new Date().toISOString()
        });
        
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

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

// Legacy endpoint for backward compatibility
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
        console.log(`🚀 Enhanced Node.js server with RAG running on port ${port}`);
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