import collections
from fastapi import FastAPI, Request, UploadFile, File, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import numpy as np
import nltk
import spacy
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from supabase import create_client, Client
import os
from os.path import join, dirname
from dotenv import load_dotenv
from datetime import datetime, timedelta
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Dict, Any, Optional
import json
from collections import Counter
import jwt
from functools import wraps
import logging
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import asyncio # Added for run_in_executor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Model Paths (TensorFlow) ---
tf_model_path_english = "deep_learning_models/final_model_english"
tf_model_path_korean = "deep_learning_models/final_model_korean"
tf_model_path_german = "deep_learning_models/final_model_german"

# --- Tokenizer Paths (TensorFlow) ---
tokenizer_path_english = "deep_learning_models/tokenizer_english"
tokenizer_path_korean = "deep_learning_models/tokenizer_korean"
tokenizer_path_german = "deep_learning_models/tokenizer_german"

# --- Global Caches for Lazy-Loaded Models ---
# Change tf_models to an OrderedDict for LRU behavior
tf_models = collections.OrderedDict()
tf_tokenizers = {} # Keep as is, or apply LRU if their memory usage is also a concern
spacy_nlp_pipelines = {} # Keep as is, or apply LRU if their memory usage is also a concern
m2m_translator_tokenizer = None
m2m_translator_model = None

# --- LRU Cache Configuration ---
# Define the maximum number of TensorFlow models to keep in cache
MAX_CACHED_MODELS = 2

# --- Helper Functions for Lazy Loading ---
def _load_tf_model_and_tokenizer(lang_code, model_path, tokenizer_path):
    # --- LRU Logic for tf_models ---
    if lang_code in tf_models:
        # If the model is already in cache, move it to the end
        # to mark it as the most recently used.
        tf_models.move_to_end(lang_code)
        logger.info(f"LRU Cache: TensorFlow {lang_code.upper()} model found in cache and marked as recently used.")
    else:
        # If the cache is full (reached MAX_CACHED_MODELS limit),
        # remove the least recently used item (from the beginning of OrderedDict).
        if len(tf_models) >= MAX_CACHED_MODELS:
            # popitem(last=False) removes the item from the beginning (LRU)
            lru_lang_code, _ = tf_models.popitem(last=False)
            logger.info(f"LRU Cache: Cache full. Unloading TensorFlow {lru_lang_code.upper()} model from memory.")
            # Note: Python's garbage collector will handle memory freeing once references are gone.

        try:
            # Load the new model and add it to the cache.
            # It will automatically be added to the end (most recently used).
            tf_models[lang_code] = tf.saved_model.load(model_path)
            logger.info(f"LRU Cache: TensorFlow {lang_code.upper()} model loaded successfully from {model_path} and added to cache.")
        except Exception as e:
            logger.error(f"LRU Cache: Failed to load TensorFlow {lang_code.upper()} model from {model_path}: {e}")
            # If loading fails, it's usually best not to cache a None or handle appropriately.
            # For simplicity, if it fails, it's not added to the cache in this block.

    # --- Tokenizer Loading (remains unchanged as per your request for models) ---
    # You could apply a similar LRU strategy to tf_tokenizers if they become a memory bottleneck.
    if lang_code not in tf_tokenizers:
        try:
            tf_tokenizers[lang_code] = AutoTokenizer.from_pretrained(tokenizer_path)
            logger.info(f"LRU Cache: {lang_code.upper()} Tokenizer loaded successfully from {tokenizer_path}.")
        except Exception as e:
            logger.error(f"LRU Cache: Failed to load {lang_code.upper()} tokenizer from {tokenizer_path}: {e}")
            tf_tokenizers[lang_code] = None
            
    # Return the loaded model and tokenizer (could be None if loading failed)
    return tf_models.get(lang_code), tf_tokenizers.get(lang_code)

def _load_spacy_nlp_pipeline(lang_code):
    """Loads and caches SpaCy NLP pipeline for a given language."""
    if lang_code not in spacy_nlp_pipelines:
        try:
            spacy_model_name = {
                'en': 'en_core_web_sm',
                'de': 'de_core_news_sm',
                'ko': 'ko_core_news_sm'
            }.get(lang_code)
            
            if not spacy_model_name:
                logger.warning(f"No SpaCy model name configured for language code: {lang_code}. Feature extraction for this language might be limited.")
                spacy_nlp_pipelines[lang_code] = None
                return None
            
            spacy_nlp_pipelines[lang_code] = spacy.load(spacy_model_name)
            logger.info(f"SpaCy '{spacy_model_name}' loaded successfully for {lang_code.upper()}.")
        except Exception as e:
            logger.error(f"Failed to load SpaCy model '{spacy_model_name}' for '{lang_code}': {e}. Please ensure it is installed (e.g., 'python -m spacy download {spacy_model_name}')")
            spacy_nlp_pipelines[lang_code] = None
    return spacy_nlp_pipelines.get(lang_code)

def _load_m2m_translator():
    """Loads and caches M2M100 translator model."""
    global m2m_translator_tokenizer, m2m_translator_model
    if m2m_translator_model is None:
        try:
            m2m_model_name = "facebook/m2m100_1.2B"
            m2m_translator_tokenizer = AutoTokenizer.from_pretrained(m2m_model_name)
            m2m_translator_model = AutoModelForSeq2SeqLM.from_pretrained(m2m_model_name)
            logger.info(f"M2M100 Translator model loaded successfully: {m2m_model_name}")
        except Exception as e:
            logger.error(f"Failed to load M2M100 Translator model: {e}")
            m2m_translator_tokenizer = None
            m2m_translator_model = None
    return m2m_translator_tokenizer, m2m_translator_model


# --- Initializations (now mostly just Supabase config and light NLP tools) ---
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

embedder = SentenceTransformer("all-MiniLM-L6-v2") 
nltk.download('punkt', quiet=False, force=True)
nltk.download('punkt_tab', quiet=False, force=True)


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Security
security = HTTPBearer(auto_error=False)

from langdetect import detect


MAX_CHUNK_LENGTH = 512
def auto_translate_to_english(text: str):
    """
    Translates text to English using M2M100 model. Lazy-loads the model.
    Uses SpaCy (if available) or regex for sentence segmentation.
    """
    translator_tokenizer_local, translator_model_local = _load_m2m_translator()
    if translator_tokenizer_local is None or translator_model_local is None:
        raise Exception("M2M100 Translator model not loaded. Cannot perform translation.")

    try:
        lang = detect(text)
        if lang == "en":
            return text, lang

        translator_tokenizer_local.src_lang = lang

        sentences = []
        spacy_for_segmentation = _load_spacy_nlp_pipeline(lang)
        if spacy_for_segmentation and spacy_for_segmentation.has_pipe('sentencizer'): # Check for sentencizer component
            doc = spacy_for_segmentation(text)
            sentences = [sent.text for sent in doc.sents]
            logger.info(f"SpaCy successfully segmented text for language: {lang}. Sentences found: {len(sentences)}")
        else:
            logger.warning(f"SpaCy pipeline for '{lang}' not available or lacks sentencizer. Falling back to basic regex segmentation for translation.")
            import re
            sentences = re.split(r'(?<=[.!?؟\u061F\u06D4])\s+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            if not sentences:
                sentences = [text.strip()] if text.strip() else []

        if not sentences:
            sentences = [text]

        translated_chunks = []
        current_chunk_sentences = []
        current_chunk_token_count = 0

        for sentence in sentences:
            sentence_tokens = translator_tokenizer_local.encode(sentence, add_special_tokens=False)
            sentence_token_count = len(sentence_tokens)

            if current_chunk_token_count + sentence_token_count + 2 > MAX_CHUNK_LENGTH and current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)

                encoded = translator_tokenizer_local(chunk_text, return_tensors="pt")
                generated = translator_model_local.generate(
                    **encoded,
                    forced_bos_token_id=translator_tokenizer_local.get_lang_id("en"),
                    max_length=MAX_CHUNK_LENGTH * 2,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3
                )
                translated_chunk = translator_tokenizer_local.batch_decode(generated, skip_special_tokens=True)[0]
                translated_chunks.append(translated_chunk)

                current_chunk_sentences = [sentence]
                current_chunk_token_count = sentence_token_count
            else:
                current_chunk_sentences.append(sentence)
                current_chunk_token_count += sentence_token_count + 2

        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            encoded = translator_tokenizer_local(chunk_text, return_tensors="pt")
            generated = translator_model_local.generate(
                **encoded,
                forced_bos_token_id=translator_tokenizer_local.get_lang_id("en"),
                max_length=MAX_CHUNK_LENGTH * 2,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
            translated_chunk = translator_tokenizer_local.batch_decode(generated, skip_special_tokens=True)[0]
            translated_chunks.append(translated_chunk)

        translated_text = " ".join(translated_chunks)
        return translated_text, lang

    except Exception as e:
        print(f"Translation error (SpaCy/regex segmentation attempt): {e}")
        return text, "en"

# --- Authentication and Pydantic Models (Unchanged) ---
async def verify_token_required(credentials: HTTPAuthorizationCredentials = Depends(security)):
    logger.info("=== REQUIRED TOKEN VERIFICATION START ===")
    if not credentials:
        logger.error("No credentials provided")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization header missing")

    try:
        token = credentials.credentials
        logger.info(f"Token received (first 20 chars): {token[:20]}...")
        try:
            logger.info("Attempting to get user with token...")
            user_response = supabase.auth.get_user(token)
            logger.info(f"Supabase response type: {type(user_response)}")
        except Exception as supabase_error:
            logger.error(f"Supabase auth error: {supabase_error}")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Supabase auth failed: {str(supabase_error)}")

        if not hasattr(user_response, 'user') or not user_response.user:
            logger.error("No user found in response")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token - no user found")

        user = user_response.user
        logger.info(f"User found: {user.email}")
        if not user.email or not user.email.endswith("@ucf.edu"):
            logger.warning(f"Non-UCF email attempted access: {user.email}")
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access restricted to UCF email addresses only")
        logger.info(f"Authentication successful for: {user.email}")
        return {"user_id": user.id, "email": user.email, "role": "authenticated"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected token verification error: {e}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Could not validate credentials: {str(e)}")

def get_current_user_required(user_data: dict = Depends(verify_token_required)):
    return user_data

class TextRequest(BaseModel):
    text: str

class BatchTextRequest(BaseModel):
    texts: List[str]

# --- Page Routes (Unchanged) ---
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    logger.info("Home page accessed")
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    logger.info("Login page accessed")
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/history", response_class=HTMLResponse)
def history(request: Request):
    logger.info("History page accessed - no backend auth required")
    return templates.TemplateResponse("history.html", {"request": request, "history": [], "user": None})

@app.get("/compare", response_class=HTMLResponse)
def compare_page(request: Request):
    logger.info("Compare page accessed - no backend auth required")
    return templates.TemplateResponse("compare.html", {"request": request, "user": None})

@app.get("/batch", response_class=HTMLResponse)
def batch_page(request: Request):
    logger.info("Batch page accessed - no backend auth required")
    return templates.TemplateResponse("batch-analyzer.html", {"request": request, "user": None})

@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request):
    logger.info("Analytics page accessed - no backend auth required")
    return templates.TemplateResponse("analytics.html", {
        "request": request,
        "user": None
    })

@app.get("/about", response_class=HTMLResponse)
def about_page(request: Request):
    logger.info("About page accessed - no backend auth required")
    return templates.TemplateResponse("about.html", {"request": request, "user": None})


# --- API Routes (get_analytics_data, get_history_data - Unchanged) ---
@app.get("/api/analytics-data")
async def get_analytics_data(current_user: dict = Depends(get_current_user_required)):
    try:
        data = supabase.table("model_data").select("*").eq("user_id", current_user["user_id"]).order("prediction_timestamp", desc=True).limit(1000).execute()
        if not data.data:
            return {"totalAnalyses": 0, "avgILR": 0.0, "languagesCount": 0, "thisWeek": 0, "ilrDistribution": [0, 0, 0, 0, 0], "languages": [], "languageCounts": [], "timeline": {"labels": [], "data": []}}
        records = data.data
        total_analyses = len(records)
        ilr_levels = [int(r.get('predicted_ilr_level', 0)) for r in records]
        avg_ilr = sum(ilr_levels) / len(ilr_levels) if ilr_levels else 0
        languages = [r.get('language', 'unknown') for r in records]
        language_counter = Counter(languages)
        top_languages = language_counter.most_common(5)
        week_ago = datetime.now() - timedelta(days=7)
        this_week = sum(1 for r in records
                       if r.get('prediction_timestamp') and
                       datetime.fromisoformat(r['prediction_timestamp'].replace('Z', '+00:00')) > week_ago)
        ilr_distribution = [0, 0, 0, 0, 0]
        for level in ilr_levels:
            if 1 <= level <= 5:
                ilr_distribution[level-1] += 1
        timeline_labels = []
        timeline_data = []
        for i in range(6):
            month_start = datetime.now().replace(day=1) - timedelta(days=30*i)
            month_name = month_start.strftime("%b")
            timeline_labels.insert(0, month_name)
            month_count = sum(1 for r in records
                            if r.get('prediction_timestamp') and
                            datetime.fromisoformat(r['prediction_timestamp'].replace('Z', '+00:00')).month == month_start.month)
            timeline_data.insert(0, month_count)
        return {
            "totalAnalyses": total_analyses,
            "avgILR": round(avg_ilr, 1),
            "languagesCount": len(language_counter),
            "thisWeek": this_week,
            "ilrDistribution": ilr_distribution,
            "languages": [lang for lang, _ in top_languages],
            "languageCounts": [count for _, count in top_languages],
            "timeline": {"labels": timeline_labels, "data": timeline_data}
        }
    except Exception as e:
        print(f"Analytics error: {e}")
        return {"totalAnalyses": 0, "avgILR": 0.0, "languagesCount": 0, "thisWeek": 0, "ilrDistribution": [0, 0, 0, 0, 0], "languages": [], "languageCounts": [], "timeline": {"labels": [], "data": []}}

@app.get("/api/history")
async def get_history_data(
    current_user: dict = Depends(get_current_user_required),
    limit: int = 50,
    offset: int = 0,
    search_query: Optional[str] = None,
    ilr_level: Optional[int] = None
):
    try:
        query = supabase.table("model_data") \
            .select("*", count='exact') \
            .eq("user_id", current_user["user_id"])

        if search_query:
            query = query.ilike("raw_text", f"%{search_query}%")

        if ilr_level is not None:
            query = query.eq("predicted_ilr_level", ilr_level)

        data = query \
            .order("prediction_timestamp", desc=True) \
            .range(offset, offset + limit - 1) \
            .execute()

        for record in data.data:
            if record.get('features') and isinstance(record['features'], str):
                try:
                    features = json.loads(record['features'])
                    cleaned_features = {}
                    for key, value in features.items():
                        if isinstance(value, float):
                            if np.isnan(value) or np.isinf(value):
                                cleaned_features[key] = None
                            else:
                                cleaned_features[key] = value
                        else:
                            cleaned_features[key] = value
                    record['features'] = cleaned_features
                except Exception as e:
                    logger.error(f"Error parsing features for record {record.get('id')}: {e}")
                    record['features'] = {}

            for key, value in record.items():
                if isinstance(value, float):
                    if np.isnan(value) or np.isinf(value):
                        record[key] = None

        return {"history": data.data, "user": current_user, "total_count": data.count}
    except Exception as e:
        logger.error(f"History API error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load history: {str(e)}")


# --- Feature Extraction (Modified for Language Specificity and Robustness) ---
def extract_features(text: str, lang_code: str):
    """
    Extracts linguistic features from text using language-specific SpaCy models.
    Loads SpaCy model on first use for a given language.
    Handles cases where certain SpaCy components (like noun_chunks, ents) are not implemented.
    """
    nlp_pipeline = _load_spacy_nlp_pipeline(lang_code)
    
    # Default values in case SpaCy model is not loaded or features are not available
    wc, avg_sent_len, readability, avg_word_len, ttr = 0, 0, 50, 0, 0
    pos_ratios = [0.0] * 5 # NOUN, VERB, ADJ, ADV, PRON
    ner_count, parse_depth, noun_chunks_count = 0, 0, 0
    embedding = embedder.encode([text])[0] # Embedding is always attempted as SentenceTransformer is multilingual

    if nlp_pipeline:
        try:
            doc = nlp_pipeline(text)
            tokens = [token.text for token in doc if not token.is_space]

            wc = len(tokens)
            sents = list(doc.sents)
            avg_sent_len = np.mean([len([t for t in sent if not t.is_space]) for sent in sents]) if sents else 0
            
            # Readability is English-specific, keep placeholder.
            # readability = textstat.flesch_reading_ease(text) if lang_code == 'en' else 50 
            
            avg_word_len = np.mean([len(t) for t in tokens]) if tokens else 0
            ttr = len(set(tokens)) / len(tokens) if tokens else 0

            pos_counts = {pos: 0 for pos in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON']}
            for token in doc:
                if token.pos_ in pos_counts:
                    pos_counts[token.pos_] += 1
            total_pos = sum(pos_counts.values()) or 1
            pos_ratios = [pos_counts[pos] / total_pos for pos in pos_counts]

            # Check if components are available before using
            if nlp_pipeline.has_pipe('ner'):
                ner_count = len(doc.ents)
            else:
                logger.warning(f"SpaCy 'ner' component not found for '{lang_code}'. NER count will be 0.")

            # Parser might not be available for all 'sm' models
            if nlp_pipeline.has_pipe('parser'): 
                parse_depth = np.mean([abs(token.head.i - token.i) for token in doc if token.head != token]) if doc else 0
            else:
                logger.warning(f"SpaCy 'parser' component not found for '{lang_code}'. Parse depth will be 0.")

            if nlp_pipeline.has_pipe('noun_chunks'): # Specific check for noun_chunks
                noun_chunks_count = len(list(doc.noun_chunks))
            else:
                logger.warning(f"SpaCy 'noun_chunks' component not found for '{lang_code}'. Noun chunks count will be 0.")

        except Exception as e:
            logger.error(f"Error during SpaCy feature extraction for '{lang_code}': {e}. Some features might be inaccurate.")
            # Keep default values assigned at the start of the function

    features_for_model = [wc, avg_sent_len, readability, avg_word_len, ttr, *pos_ratios, ner_count, parse_depth, noun_chunks_count, *embedding]
    
    features_dict = {
        "Word Count": wc,
        "Avg Sentence Length": avg_sent_len,
        "Readability Score": readability,
        "Avg Word Length": avg_word_len,
        "Type-Token Ratio": ttr,
        "Noun Ratio": pos_ratios[0],
        "Verb Ratio": pos_ratios[1],
        "Adj Ratio": pos_ratios[2],
        "Adv Ratio": pos_ratios[3],
        "Pronoun Ratio": pos_ratios[4],
        "NER Count": ner_count,
        "Parse Depth": parse_depth,
        "Noun Chunks": noun_chunks_count,
    }
    
    return np.array(features_for_model).reshape(1, -1), features_dict

def normalize_features(features_dict):
    max_vals = {
        "Word Count": 500,
        "Avg Sentence Length": 40,
        "Readability Score": 100,
        "Avg Word Length": 10,
        "Type-Token Ratio": 1,
        "Noun Ratio": 1,
        "Verb Ratio": 1,
        "Adj Ratio": 1,
        "Adv Ratio": 1,
        "Pronoun Ratio": 1,
        "NER Count": 30,
        "Parse Depth": 20,
        "Noun Chunks": 250,
    }

    normalized_dict = {}
    for k, v in features_dict.items():
        if k.startswith("Probabilities_ILR_"): # Do not normalize probabilities, pass them through
            normalized_dict[k] = v
        elif k in max_vals and max_vals[k] != 0:
            normalized_dict[k] = min(v / max_vals[k], 1.0)
        else:
            normalized_dict[k] = v # Keep other features as is if not in max_vals or not a number
    return normalized_dict

# --- Store Prediction (Unchanged, only called by background task) ---
def store_prediction(raw_text, translated_text, language, predicted_ilr, features_dict=None, user_id=None):
    try:
        existing = supabase.table("text_data") \
            .select("id") \
            .eq("raw_text", raw_text) \
            .eq("language", language) \
            .eq("user_id", user_id) \
            .limit(1) \
            .execute()

        text_id = None
        if existing.data:
            text_id = existing.data[0]["id"]
        else:
            insert_res = supabase.table("text_data").insert({
                "raw_text": raw_text,
                "translated_text": translated_text,
                "language": language,
                "ilr_level": predicted_ilr,
                "user_id": user_id
            }).execute()
            text_id = insert_res.data[0]["id"]

        adjusted_time = (datetime.utcnow() - timedelta(hours=4)).isoformat()
        model_data = {
            "source_text_id": text_id,
            "raw_text": raw_text,
            "translated_text": translated_text,
            "language": language,
            "predicted_ilr_level": predicted_ilr,
            "prediction_timestamp": adjusted_time,
            "user_id": user_id
        }

        if features_dict:
            cleaned_features = {}
            for key, value in features_dict.items():
                if isinstance(value, float):
                    if np.isnan(value) or np.isinf(value):
                        cleaned_features[key] = 0.0
                    else:
                        cleaned_features[key] = value
                else:
                    cleaned_features[key] = value
            model_data["features"] = json.dumps(cleaned_features)

        supabase.table("model_data").insert(model_data).execute()

    except Exception as e:
        print(f"Storage error: {e}")

# --- Background Task (Modified to receive all data for storage) ---
def store_prediction_background(raw_text, translated_text, detected_lang, predicted_ilr, user_id, features_dict):
    """
    Function to perform database storage in the background.
    It receives all necessary data from the foreground process.
    """
    logger.info(f"Starting background DB storage for text: {raw_text[:50]}...")
    try:
        store_prediction(
            raw_text=raw_text,
            translated_text=translated_text,
            language=detected_lang,
            predicted_ilr=predicted_ilr,
            features_dict=features_dict, # Use features_dict already processed in foreground
            user_id=user_id
        )
        logger.info(f"Background DB storage completed for text: {raw_text[:50]}...")
    except Exception as e:
        logger.exception(f"Background DB storage failed for text: {raw_text[:50]} with error: {e}")

# --- TensorFlow Prediction Core Logic (Unchanged) ---
def predict_with_tf_model(text: str, model_tf, tokenizer_tf):
    if model_tf is None or tokenizer_tf is None:
        raise Exception("TensorFlow model or tokenizer not loaded for the specified language.")

    max_chunk_length = 256
    num_expected_chunks = 5
    # Define the exact max length for the tokenizer to pad/truncate to
    desired_flat_length = num_expected_chunks * max_chunk_length # This will be 1280

    tokenized_output = tokenizer_tf(
        text,
        return_tensors="tf",
        padding="max_length", # Pad to `max_length`
        truncation=True,      # Crucially, set to True to truncate texts that are too long
        max_length=desired_flat_length # Pad/truncate to exactly 1280 tokens
    )

    input_ids_flat = tokenized_output['input_ids'][0]
    attention_mask_flat = tokenized_output['attention_mask'][0]

    # After this tokenizer call, input_ids_flat and attention_mask_flat
    # should ALWAYS have a length of `desired_flat_length` (1280).
    # Therefore, your manual padding/truncation logic is no longer needed.

    # Now, reshape directly
    input_ids_reshaped = tf.reshape(input_ids_flat, [1, num_expected_chunks, max_chunk_length])
    attention_mask_reshaped = tf.reshape(attention_mask_flat, [1, num_expected_chunks, max_chunk_length])

    model_inputs = {
        'input_ids': input_ids_reshaped,
        'attention_mask': attention_mask_reshaped
    }

    outputs = model_tf(model_inputs, training=False, mask=None)

    predicted_ilr = 0
    probabilities = None

    try:
        output_tensor = None
        if hasattr(outputs, 'logits'):
            output_tensor = outputs.logits
        elif isinstance(outputs, (tf.Tensor, np.ndarray)):
            output_tensor = outputs
        elif isinstance(outputs, dict) and 'predictions' in outputs:
            output_tensor = outputs['predictions']
        else:
            logger.warning(f"Unexpected TensorFlow model output type: {type(outputs)}. Cannot extract probabilities reliably.")
            return predicted_ilr, [0.2, 0.2, 0.2, 0.2, 0.2]

        if output_tensor.shape.rank == 1:
            output_tensor = tf.expand_dims(output_tensor, axis=0)

        probabilities = output_tensor.numpy().flatten()

        predicted_class_id_tensor = tf.argmax(output_tensor, axis=1)
        predicted_class_id = int(predicted_class_id_tensor.numpy().flatten()[0])

        predicted_ilr = int(predicted_class_id) # Assuming model outputs 0-4 for ILR 1-5

    except Exception as e_extract:
        logger.error(f"Error extracting prediction or probabilities from model output: {e_extract}. Defaulting to ILR 2 and uniform probabilities.")
        predicted_ilr = 2 # Default to a mid-range ILR if error
        probabilities = [0.2, 0.2, 0.2, 0.2, 0.2]

    return predicted_ilr, probabilities.tolist()

# --- Main Prediction Endpoints (Modified for New Logic) ---
@app.post("/predict")
async def predict_ilr(request: TextRequest, background_tasks: BackgroundTasks, current_user: dict = Depends(get_current_user_required)):
    logger.info(f"Predict request from user: {current_user.get('email', 'unknown')}")
    raw_text = request.text
    logger.info(f"Processing text of length: {len(raw_text)}")

    predicted_ilr = 2
    detected_lang = "en"
    probabilities = [0.2, 0.2, 0.2, 0.2, 0.2] # Default to uniform
    translated_text = None # Will be populated if translation occurs
    features_dict = {}
    normalized_features = {}

    try:
        detected_lang = detect(raw_text)
    except Exception as e:
        logger.warning(f"Language detection failed: {e}. Defaulting to 'en'.")
        detected_lang = "en"

    tf_model_to_use, tf_tokenizer_to_use = None, None
    text_for_tf_prediction = raw_text 
    lang_for_feature_extraction = detected_lang 

    loop = asyncio.get_event_loop() 

    if detected_lang == 'en':
        tf_model_to_use, tf_tokenizer_to_use = _load_tf_model_and_tokenizer('en', tf_model_path_english, tokenizer_path_english)
    elif detected_lang == 'ko':
        tf_model_to_use, tf_tokenizer_to_use = _load_tf_model_and_tokenizer('ko', tf_model_path_korean, tokenizer_path_korean)
    elif detected_lang == 'de':
        tf_model_to_use, tf_tokenizer_to_use = _load_tf_model_and_tokenizer('de', tf_model_path_german, tokenizer_path_german)

    if tf_model_to_use and tf_tokenizer_to_use:
        try:
   
            predicted_ilr, probabilities = await loop.run_in_executor(
                None, 
                predict_with_tf_model, 
                text_for_tf_prediction, tf_model_to_use, tf_tokenizer_to_use 
            )
            logger.info(f"Prediction made using native TF model ({detected_lang}). ILR: {predicted_ilr}, Probs: {probabilities}")
        except Exception as e:
            logger.error(f"Native TF model prediction for {detected_lang} failed: {e}. Attempting English TF model fallback.")
        
            tf_model_to_use, tf_tokenizer_to_use = _load_tf_model_and_tokenizer('en', tf_model_path_english, tokenizer_path_english)
            if tf_model_to_use and tf_tokenizer_to_use:
           
                translated_text, _ = await loop.run_in_executor(
                    None,
                    auto_translate_to_english,
                    raw_text
                )
                text_for_tf_prediction = translated_text
                lang_for_feature_extraction = 'en'
         
                predicted_ilr, probabilities = await loop.run_in_executor(
                    None,
                    predict_with_tf_model,
                    text_for_tf_prediction, tf_model_to_use, tf_tokenizer_to_use
                )
                logger.info(f"Prediction made using English TF model (fallback). ILR: {predicted_ilr}, Probs: {probabilities}")
            else:
                logger.error("English TF model not available for fallback. Returning default prediction.")
    else: 
        logger.warning(f"No native TF model for '{detected_lang}'. Translating and using English TF model.")
        tf_model_to_use, tf_tokenizer_to_use = _load_tf_model_and_tokenizer('en', tf_model_path_english, tokenizer_path_english)
        if tf_model_to_use and tf_tokenizer_to_use:
      
            translated_text, _ = await loop.run_in_executor(
                None,
                auto_translate_to_english,
                raw_text
            )
            text_for_tf_prediction = translated_text
            lang_for_feature_extraction = 'en'

            predicted_ilr, probabilities = await loop.run_in_executor(
                None,
                predict_with_tf_model,
                text_for_tf_prediction, tf_model_to_use, tf_tokenizer_to_use
            )
            logger.info(f"Prediction made using English TF model (fallback). ILR: {predicted_ilr}, Probs: {probabilities}")
        else:
            logger.error("English TF model not available for fallback. Returning default prediction.")

    try:
        _, features_dict = await loop.run_in_executor(
            None,
            extract_features,
            text_for_tf_prediction, lang_for_feature_extraction
        )
     
        if probabilities:
            for idx, prob in enumerate(probabilities):
                features_dict[f'Probabilities_ILR_{idx}'] = prob
        normalized_features = normalize_features(features_dict)
    except Exception as e:
        logger.error(f"Feature extraction failed for {lang_for_feature_extraction} text: {e}. Returning empty features.")
        features_dict = {}
        normalized_features = {}

    background_tasks.add_task(
        store_prediction_background,
        raw_text=raw_text,
        translated_text=translated_text, 
        detected_lang=detected_lang,
        predicted_ilr=predicted_ilr,
        user_id=current_user["user_id"],
        features_dict=features_dict 
    )

    # --- IMMEDIATE RESPONSE TO FRONTEND ---
    return {
        "predicted_ilr": int(predicted_ilr),
        "probabilities": probabilities,
        "features": normalized_features, # Normalized linguistic features + probabilities
        "raw_features": features_dict, # Raw linguistic features + probabilities
        "original_language": detected_lang,
        "translated_text": translated_text # Sent to frontend if translation occurred
    }

@app.post("/predict-batch")
async def predict_batch(request: BatchTextRequest, background_tasks: BackgroundTasks, current_user: dict = Depends(get_current_user_required)):
    # Batch prediction now performs all steps in foreground per item, like /predict
    try:
        results = []
        loop = asyncio.get_event_loop() # Get the event loop here for batch processing

        for i, text in enumerate(request.texts):
            if not text.strip():
                continue

            predicted_ilr_for_item = 2
            detected_lang_for_item = "en"
            probabilities_for_item = [0.2, 0.2, 0.2, 0.2, 0.2] # Default
            translated_text_for_item = None
            features_dict_for_item = {}
            normalized_features_for_item = {}

            try:
                detected_lang_for_item = detect(text)
            except Exception as e:
                logger.warning(f"Language detection failed for text {i}: {e}. Defaulting to 'en'.")
                detected_lang_for_item = "en"

            # --- Prediction (Native TF or Translated-to-English TF) ---
            tf_model_to_use, tf_tokenizer_to_use = None, None
            if detected_lang_for_item == 'en':
                tf_model_to_use, tf_tokenizer_to_use = _load_tf_model_and_tokenizer('en', tf_model_path_english, tokenizer_path_english)
            elif detected_lang_for_item == 'ko':
                tf_model_to_use, tf_tokenizer_to_use = _load_tf_model_and_tokenizer('ko', tf_model_path_korean, tokenizer_path_korean)
            elif detected_lang_for_item == 'de':
                tf_model_to_use, tf_tokenizer_to_use = _load_tf_model_and_tokenizer('de', tf_model_path_german, tokenizer_path_german)

            text_for_tf_prediction = text # Initialize with original text
            lang_for_feature_extraction = detected_lang_for_item # Initialize language for feature extraction

            if tf_model_to_use and tf_tokenizer_to_use:
                try:
                    # Offload prediction with native model
                    predicted_ilr_for_item, probabilities_for_item = await loop.run_in_executor(
                        None,
                        predict_with_tf_model,
                        text_for_tf_prediction, tf_model_to_use, tf_tokenizer_to_use
                    )
                    logger.info(f"Text {i}: Prediction made using native TF model ({detected_lang_for_item}). ILR: {predicted_ilr_for_item}")
                except Exception as e:
                    logger.error(f"Text {i}: Native TF model prediction for {detected_lang_for_item} failed: {e}. Attempting English TF model fallback.")
                    tf_model_to_use, tf_tokenizer_to_use = _load_tf_model_and_tokenizer('en', tf_model_path_english, tokenizer_path_english)
                    if tf_model_to_use and tf_tokenizer_to_use:
                        # Offload translation
                        translated_text_for_item, _ = await loop.run_in_executor(
                            None,
                            auto_translate_to_english,
                            text
                        )
                        text_for_tf_prediction = translated_text_for_item
                        lang_for_feature_extraction = 'en'
                        # Offload prediction with English model
                        predicted_ilr_for_item, probabilities_for_item = await loop.run_in_executor(
                            None,
                            predict_with_tf_model,
                            text_for_tf_prediction, tf_model_to_use, tf_tokenizer_to_use
                        )
                        logger.info(f"Text {i}: Prediction made using English TF model (fallback). ILR: {predicted_ilr_for_item}")
                    else:
                        logger.error(f"Text {i}: English TF model not available for fallback. Returning default prediction and empty features.")
            else: # No native TF model available for detected_lang
                logger.warning(f"Text {i}: No native TF model for '{detected_lang_for_item}'. Translating and using English TF model.")
                tf_model_to_use, tf_tokenizer_to_use = _load_tf_model_and_tokenizer('en', tf_model_path_english, tokenizer_path_english)
                if tf_model_to_use and tf_tokenizer_to_use:
                    # Offload translation
                    translated_text_for_item, _ = await loop.run_in_executor(
                        None,
                        auto_translate_to_english,
                        text
                    )
                    text_for_tf_prediction = translated_text_for_item
                    lang_for_feature_extraction = 'en'
                    # Offload prediction with English model
                    predicted_ilr_for_item, probabilities_for_item = await loop.run_in_executor(
                        None,
                        predict_with_tf_model,
                        text_for_tf_prediction, tf_model_to_use, tf_tokenizer_to_use
                    )
                    logger.info(f"Text {i}: Prediction made using English TF model (fallback). ILR: {predicted_ilr_for_item}")
                else:
                    logger.error(f"Text {i}: English TF model not available for fallback. Returning default prediction and empty features.")

            # --- Feature Extraction for item ---
            try:
                _, features_dict_for_item = await loop.run_in_executor(
                    None,
                    extract_features,
                    text_for_tf_prediction, lang_for_feature_extraction
                )
                if probabilities_for_item:
                    for idx, prob in enumerate(probabilities_for_item):
                        features_dict_for_item[f'Probabilities_ILR_{idx}'] = prob
                normalized_features_for_item = normalize_features(features_dict_for_item)
            except Exception as e:
                logger.error(f"Text {i}: Feature extraction failed for {lang_for_feature_extraction} text: {e}. Returning empty features.")
                features_dict_for_item = {}
                normalized_features_for_item = {}

            # --- Background Storage for item ---
            background_tasks.add_task(
                store_prediction_background,
                raw_text=text,
                translated_text=translated_text_for_item,
                detected_lang=detected_lang_for_item,
                predicted_ilr=predicted_ilr_for_item,
                user_id=current_user["user_id"],
                features_dict=features_dict_for_item
            )

            result = {
                "index": i,
                "text": text,
                "predicted_ilr": int(predicted_ilr_for_item),
                "probabilities": probabilities_for_item,
                "features": normalized_features_for_item,
                "raw_features": features_dict_for_item,
                "original_language": detected_lang_for_item,
                "translated_text": translated_text_for_item
            }
            results.append(result)

        return {
            "results": results,
            "total_processed": len(results),
            "success_count": len([r for r in results if "error" not in r])
        }

    except Exception as e:
        logger.exception(f"Batch prediction error: {e}")
        return {
            "results": [],
            "total_processed": 0,
            "success_count": 0,
            "error": str(e)
        }

@app.post("/upload-files")
async def upload_files(files: List[UploadFile] = File(...), current_user: dict = Depends(get_current_user_required)):
    try:
        texts = []
        for file in files:
            if file.content_type == "text/plain":
                content = await file.read()
                text = content.decode("utf-8")
                texts.append({"filename": file.filename, "content": text})
        return {"texts": texts}
    except Exception as e:
        print(f"File upload error: {e}")
        return {"texts": [], "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)