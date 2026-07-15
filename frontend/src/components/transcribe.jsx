import React, { useState, useRef, useEffect, useMemo, useCallback } from 'react';

// ---------------------------------------------------------------------------
// Mock data — shaped exactly like the real /transcribe response.
// Swap MOCK_RESPONSE for a live fetch() to http://127.0.0.1:8000/transcribe
// ---------------------------------------------------------------------------
const MOCK_RESPONSE = {
    language: 'en',
    processing_time_seconds: 21.38,
    speech_duration_seconds: 70.87,
    vad_enabled: true,
    noise_removal_enabled: true,
    timestamp: '20260714_151652',
    segments: [
        { start: 2.29, end: 4.97, text: "one eight two heavy your two and a half miles" },
        { start: 4.97, end: 7.66, text: "from lower cross lower three thousand feet above cleared ils" },
        { start: 7.66, end: 10.34, text: "dme approach runway two seven maintain one hundred seventy knots" },
        { start: 10.34, end: 13.03, text: "to rapet please ok confirm cleared ils cleared ils dme" },
        { start: 13.03, end: 15.71, text: "approach runway two seven transportugal three two two heavy runway" },
        { start: 15.71, end: 18.4, text: "two seven ok thank you cleared ils to seven portugal" },
        { start: 18.4, end: 21.08, text: "three two heavy next eurotrans four five your miles from" },
        { start: 21.08, end: 23.77, text: "wind cross wind a three thousand cleared ils two two" },
        { start: 23.77, end: 26.45, text: "left approach hold short of runway two seven maintain speed" },
        { start: 26.45, end: 29.13, text: "one seven zero till lindy contract all ok we'll do" },
        { start: 29.13, end: 31.82, text: "it one eight four five american seven thirty six traffic" },
        { start: 31.82, end: 34.5, text: "landing two two left or hold short of your runway" },
        { start: 34.5, end: 37.19, text: "contact tower one one niner point one good day maintain" },
        { start: 37.19, end: 39.87, text: "one good day final ten seventy two ils four eight" },
        { start: 39.87, end: 42.56, text: "for five thousand kilo ten seventy two thanks ten eighty" },
        { start: 42.56, end: 45.24, text: "present heading maintain four thousand kilo ten eighty present heading" },
        { start: 45.24, end: 47.93, text: "descend to maintain four thousand descend to four thousand ten" },
        { start: 47.93, end: 50.61, text: "eighty next five hundred tower one one niner point one" },
        { start: 53.15, end: 56.01, text: "speedbird niner fifteen fly heading one niner zero and intercept" },
        { start: 56.01, end: 58.86, text: "localizer descend to maintain three thousand one niner zero intercept" },
        { start: 58.86, end: 61.15, text: "localizer down to three thousand feet niner fifteen" },
        { start: 62.79, end: 65.24, text: "and ciao twenty one twelve is with you at one" },
        { start: 65.24, end: 67.69, text: "point five cleared twenty one twelve roger proceed minus or" },
        { start: 67.69, end: 70.14, text: "greater runway two seven roger ten eighty turn left heading" },
        { start: 70.14, end: 70.87, text: "three six zero" },
    ],
    intermediate_chunks: [
        { chunk_num: 1, start_time: 0.0, end_time: 30.0 },
        { chunk_num: 2, start_time: 25.0, end_time: 55.0 },
        { chunk_num: 3, start_time: 50.0, end_time: 80.0 },
        { chunk_num: 4, start_time: 75.0, end_time: 105.0 },
        { chunk_num: 5, start_time: 100.0, end_time: 130.0 },
        { chunk_num: 6, start_time: 125.0, end_time: 155.0 },
        { chunk_num: 7, start_time: 150.0, end_time: 180.0 },
        { chunk_num: 8, start_time: 175.0, end_time: 205.0 },
    ],
};
MOCK_RESPONSE.filtered_full_path_rel = `output/filtered_full_${MOCK_RESPONSE.timestamp}.wav`;
MOCK_RESPONSE.denoised_full_path_rel = `output/denoised_full_${MOCK_RESPONSE.timestamp}.wav`;
MOCK_RESPONSE.device = 'cuda';
MOCK_RESPONSE.intermediate_chunks = Array.from({ length: 24 }, (_, i) => {
    const chunk_num = i + 1;
    const start_time = i * 25.0;
    const end_time = Math.min(600.0, start_time + 30.0);
    return {
        chunk_num,
        start_time,
        end_time,
        filtered_path_rel: `output/filtered_chunk_${chunk_num}_${MOCK_RESPONSE.timestamp}.wav`,
        denoised_path_rel: `output/denoised_chunk_${chunk_num}_${MOCK_RESPONSE.timestamp}.wav`,
    };
});
MOCK_RESPONSE.text = MOCK_RESPONSE.segments.map((s) => s.text).join(' ');

// ---------------------------------------------------------------------------
// ATC phraseology tagger — flags callsigns, altitudes, headings, runways,
// clearance verbs so the transcript reads like a strip, not a wall of text.
// ---------------------------------------------------------------------------
const PATTERNS = [
    { re: /\b(cleared|maintain|contact|descend|climb|turn left|turn right|hold short|reduce speed)\b/gi, cls: 'tok-verb' },
    { re: /\brunway two\s(seven|two left|two)\b/gi, cls: 'tok-runway' },
    { re: /\b(one|two|three|four|five|six|seven|eight|niner|nine|zero)\s(hundred|thousand)\b/gi, cls: 'tok-alt' },
    { re: /\bheading\s(one|two|three|zero|niner|nine)?\s?[\w\s]{0,12}?(zero|one|two|three|four|five|six|seven|eight|niner|nine)\b/gi, cls: 'tok-heading' },
    { re: /\b(ils|dme|localizer|qnh|vfr)\b/gi, cls: 'tok-nav' },
];

function tagText(text) {
    // Build a list of non-overlapping matches, first-match-wins by pattern order.
    const marks = new Array(text.length).fill(null);
    PATTERNS.forEach(({ re, cls }) => {
        let m;
        const rx = new RegExp(re.source, re.flags);
        while ((m = rx.exec(text)) !== null) {
            if (m[0].length === 0) { rx.lastIndex++; continue; }
            let clash = false;
            for (let i = m.index; i < m.index + m[0].length; i++) if (marks[i]) { clash = true; break; }
            if (!clash) for (let i = m.index; i < m.index + m[0].length; i++) marks[i] = cls;
        }
    });
    const nodes = [];
    let i = 0;
    while (i < text.length) {
        if (!marks[i]) {
            let j = i;
            while (j < text.length && !marks[j]) j++;
            nodes.push({ t: text.slice(i, j), cls: null });
            i = j;
        } else {
            const cls = marks[i];
            let j = i;
            while (j < text.length && marks[j] === cls) j++;
            nodes.push({ t: text.slice(i, j), cls });
            i = j;
        }
    }
    return nodes;
}

function fmtTime(s) {
    const m = Math.floor(s / 60);
    const sec = (s % 60).toFixed(1).padStart(4, '0');
    return `${String(m).padStart(2, '0')}:${sec}`;
}

// ---------------------------------------------------------------------------
// Synthetic audio buffer generator — stands in for the real chunk WAVs
// (which live at local D:\ paths, not fetchable URLs). Each chunk gets a
// distinct tone + light noise so filtered vs denoised are audibly different.
// Swap generateChunkBuffer() for a real fetch(chunk.denoised_path_rel) once
// your backend serves these statically.
// ---------------------------------------------------------------------------
function generateChunkBuffer(ctx, chunkNum, variant, durationSec) {
    const sr = ctx.sampleRate;
    const len = Math.floor(sr * durationSec);
    const buf = ctx.createBuffer(1, len, sr);
    const data = buf.getChannelData(0);
    const baseFreq = 180 + (chunkNum % 8) * 35;
    const noiseAmt = variant === 'filtered' ? 0.22 : 0.05;
    for (let i = 0; i < len; i++) {
        const t = i / sr;
        const tone = Math.sin(2 * Math.PI * baseFreq * t) * 0.15
            + Math.sin(2 * Math.PI * (baseFreq * 2.01) * t) * 0.05;
        const noise = (Math.random() * 2 - 1) * noiseAmt;
        const env = Math.min(1, t * 8) * Math.min(1, (durationSec - t) * 8 + 0.05);
        data[i] = (tone + noise) * Math.max(0, env) * 0.5;
    }
    return buf;
}

// ---------------------------------------------------------------------------
export default function ATCConsole() {
    const [dataset, setDataset] = useState(MOCK_RESPONSE);
    const data = dataset;
    const totalDuration = data.speech_duration_seconds || 1;
    const speedup = data.processing_time_seconds ? (data.speech_duration_seconds / data.processing_time_seconds).toFixed(1) : '0';

    const [playhead, setPlayhead] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [activeSegIdx, setActiveSegIdx] = useState(-1);
    const [viewMode, setViewMode] = useState('segments'); // segments | full
    const [selectedChunk, setSelectedChunk] = useState(data.intermediate_chunks[0]?.chunk_num || 1);
    const [chunkVariant, setChunkVariant] = useState('denoised'); // filtered | denoised
    const [chunkPlaying, setChunkPlaying] = useState(false);
    const [chunkLoading, setChunkLoading] = useState(false);
    const [fullVariant, setFullVariant] = useState('denoised'); // filtered | denoised

    // State for transcription history list
    const [historyList, setHistoryList] = useState([]);
    const [historyPage, setHistoryPage] = useState(1);
    const [historyTotalPages, setHistoryTotalPages] = useState(1);
    const [historyTotal, setHistoryTotal] = useState(0);
    const [historyLimit] = useState(5);
    const [historyLoading, setHistoryLoading] = useState(false);
    const [loadedHistoryId, setLoadedHistoryId] = useState(null);

    // State for transcription form
    const [file, setFile] = useState(null);
    const [applyNoiseRemoval, setApplyNoiseRemoval] = useState(true);
    const [noiseReduction, setNoiseReduction] = useState(0.75);
    const transcribeLang = 'en'; // English constant
    const [transcribeDevice, setTranscribeDevice] = useState('cuda'); // cuda | cpu
    const [saveIntermediate, setSaveIntermediate] = useState(true);
    const [isTranscribing, setIsTranscribing] = useState(false);
    const [transcribeError, setTranscribeError] = useState(null);
    const [statusMessage, setStatusMessage] = useState('');

    const audioCtxRef = useRef(null);
    const mainRafRef = useRef(null);
    const mainStartRef = useRef(0);
    const chunkSourceRef = useRef(null);
    const segRefs = useRef({});
    const canvasRef = useRef(null);
    const audioElRef = useRef(null);
    const segListRef = useRef(null);

    if (!audioElRef.current) {
        audioElRef.current = new Audio();
    }

    const getCtx = useCallback(() => {
        if (!audioCtxRef.current) {
            audioCtxRef.current = new (window.AudioContext || window.webkitAudioContext)();
        }
        return audioCtxRef.current;
    }, []);

    const fetchHistory = useCallback(async (page = 1) => {
        setHistoryLoading(true);
        try {
            const res = await fetch(`/transcriptions?page=${page}&limit=${historyLimit}`);
            if (!res.ok) throw new Error(`HTTP error: ${res.status}`);
            const data = await res.json();
            setHistoryList(data.transcriptions || []);
            setHistoryPage(data.page || 1);
            setHistoryTotal(data.total || 0);
            setHistoryTotalPages(data.pages || 1);
            console.log("Loaded previous transcriptions list in console:", data.transcriptions);
        } catch (err) {
            console.error("Failed to fetch transcription history:", err);
        } finally {
            setHistoryLoading(false);
        }
    }, [historyLimit]);

    const loadHistoryItem = useCallback(async (id) => {
        try {
            const res = await fetch(`/transcriptions/${id}`);
            if (!res.ok) throw new Error(`HTTP error: ${res.status}`);
            const detail = await res.json();
            
            if (detail && detail.segments) {
                if (!detail.text) {
                    detail.text = detail.segments.map(s => s.text).join(' ');
                }
                
                // Stop any playing audio before changing dataset
                if (audioElRef.current) {
                    audioElRef.current.pause();
                }
                setIsPlaying(false);
                if (chunkSourceRef.current) {
                    try { chunkSourceRef.current.stop(); } catch (e) { }
                }
                setChunkPlaying(false);
                
                setDataset(detail);
                setLoadedHistoryId(detail.id);
                setPlayhead(0);
                if (detail.intermediate_chunks && detail.intermediate_chunks.length > 0) {
                    setSelectedChunk(detail.intermediate_chunks[0].chunk_num);
                }
                setStatusMessage(`Loaded transcription ID ${detail.id}: ${detail.filename}`);
                console.log(`Successfully loaded historical transcription details for ID ${detail.id}:`, detail);
            }
        } catch (err) {
            console.error(`Failed to load transcription detail for ID ${id}:`, err);
            setStatusMessage(`Failed to load historical transcription: ${err.message}`);
        }
    }, []);

    // Load transcription history on mount
    useEffect(() => {
        fetchHistory(1).then(() => {
            // Fetch default mock transcription JSON
            fetch('/output/segment_03_30-40min_transcription_20260714_151652.json')
                .then(res => {
                    if (!res.ok) throw new Error("Mock transcription JSON not found");
                    return res.json();
                })
                .then(jsonData => {
                    if (jsonData && jsonData.segments) {
                        if (!jsonData.text) {
                            jsonData.text = jsonData.segments.map(s => s.text).join(' ');
                        }
                        setDataset(jsonData);
                    }
                })
                .catch(err => {
                    console.log("Could not load full transcription JSON, using default truncated mock data:", err);
                });
        });
    }, [fetchHistory]);

    // Sync selected chunk if dataset changes
    useEffect(() => {
        if (data.intermediate_chunks && data.intermediate_chunks.length > 0) {
            if (!data.intermediate_chunks.some(c => c.chunk_num === selectedChunk)) {
                setSelectedChunk(data.intermediate_chunks[0].chunk_num);
            }
        }
    }, [data.intermediate_chunks]);

    // Handle audio element events
    useEffect(() => {
        const audio = audioElRef.current;
        if (!audio) return;
        
        const handleTimeUpdate = () => {
            if (!audio.paused) {
                setPlayhead(audio.currentTime);
            }
        };
        
        const handleEnded = () => {
            setIsPlaying(false);
            setPlayhead(0);
        };
        
        audio.addEventListener('timeupdate', handleTimeUpdate);
        audio.addEventListener('ended', handleEnded);
        
        return () => {
            audio.removeEventListener('timeupdate', handleTimeUpdate);
            audio.removeEventListener('ended', handleEnded);
            audio.pause();
            audio.src = '';
        };
    }, []);

    // Sync playhead when variant or dataset changes
    useEffect(() => {
        const audio = audioElRef.current;
        if (!audio) return;
        
        const audioSrc = fullVariant === 'denoised' ? data.denoised_full_path_rel : data.filtered_full_path_rel;
        const audioUrl = audioSrc ? `/${audioSrc}` : '';
        
        if (audioUrl && (!audio.src || !audio.src.includes(audioUrl))) {
            const wasPlaying = isPlaying;
            if (wasPlaying) audio.pause();
            audio.src = audioUrl;
            audio.load();
            audio.currentTime = playhead;
            if (wasPlaying) {
                audio.play().catch(e => console.log("Audio play failed on variant change:", e));
            }
        }
    }, [fullVariant, data.denoised_full_path_rel, data.filtered_full_path_rel]);

    // ---- main "waveform" playhead simulation fallback (drives segment sync if audio is not playing) ----
    useEffect(() => {
        if (!isPlaying) {
            if (mainRafRef.current) cancelAnimationFrame(mainRafRef.current);
            return;
        }

        const audio = audioElRef.current;
        const usingRealAudio = audio && audio.src && !audio.paused && !audio.ended;
        if (usingRealAudio) {
            return;
        }

        mainStartRef.current = performance.now() - playhead * 1000;
        const tick = () => {
            const elapsed = (performance.now() - mainStartRef.current) / 1000;
            if (elapsed >= totalDuration) {
                setPlayhead(totalDuration);
                setIsPlaying(false);
                return;
            }
            setPlayhead(elapsed);
            mainRafRef.current = requestAnimationFrame(tick);
        };
        mainRafRef.current = requestAnimationFrame(tick);
        return () => cancelAnimationFrame(mainRafRef.current);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [isPlaying, totalDuration]);

    useEffect(() => {
        if (!data.segments || data.segments.length === 0) return;
        
        // Find the active segment, or the most recent one if in a gap
        let idx = data.segments.findIndex((s) => playhead >= s.start && playhead <= s.end);
        if (idx === -1) {
            idx = data.segments.reduce((bestIdx, s, currIdx) => {
                if (s.start <= playhead) {
                    return currIdx;
                }
                return bestIdx;
            }, -1);
        }
        
        setActiveSegIdx(idx);
    }, [playhead, data.segments]);

    // Handle scroll alignment separately when activeSegIdx changes
    useEffect(() => {
        if (activeSegIdx >= 0 && segRefs.current[activeSegIdx] && segListRef.current) {
            const container = segListRef.current;
            const element = segRefs.current[activeSegIdx];
            const elemTop = element.offsetTop;
            
            container.scrollTo({
                top: elemTop - container.clientHeight / 2 + element.offsetHeight / 2,
                behavior: 'smooth'
            });
        }
    }, [activeSegIdx]);

    const seekTo = (t) => {
        setPlayhead(t);
        const audio = audioElRef.current;
        if (audio && audio.src) {
            audio.currentTime = t;
        }
        mainStartRef.current = performance.now() - t * 1000;
    };

    const togglePlay = () => {
        stopChunk();

        const audio = audioElRef.current;
        if (!audio) return;
        
        const audioSrc = fullVariant === 'denoised' ? data.denoised_full_path_rel : data.filtered_full_path_rel;
        const audioUrl = audioSrc ? `/${audioSrc}` : '';
        
        if (audioUrl && (!audio.src || !audio.src.includes(audioUrl))) {
            audio.src = audioUrl;
            audio.load();
            audio.currentTime = playhead;
        }

        if (isPlaying) {
            audio.pause();
            setIsPlaying(false);
        } else {
            audio.play()
                .then(() => setIsPlaying(true))
                .catch(e => {
                    console.error("Playback failed, starting simulation fallback:", e);
                    setIsPlaying(true);
                });
        }
    };

    // ---- transcription form submission ----
    const handleTranscribe = async (e) => {
        e.preventDefault();
        if (!file) return;

        setIsTranscribing(true);
        setTranscribeError(null);
        setStatusMessage('Uploading and processing audio file...');
        
        if (isPlaying) {
            audioElRef.current.pause();
            setIsPlaying(false);
        }
        stopChunk();

        const formData = new FormData();
        formData.append('file', file);
        formData.append('apply_noise_removal', applyNoiseRemoval.toString());
        formData.append('noise_reduction', noiseReduction.toString());
        formData.append('language', transcribeLang);
        formData.append('save_intermediate', saveIntermediate.toString());
        formData.append('device', transcribeDevice);

        try {
            const response = await fetch('/transcribe', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || `Server error: ${response.status}`);
            }

            const result = await response.json();
            
            if (result && result.segments) {
                if (!result.text) {
                    result.text = result.segments.map(s => s.text).join(' ');
                }
                
                // If backend didn't generate intermediate chunks info or it is empty
                if (!result.intermediate_chunks || result.intermediate_chunks.length === 0) {
                    const dur = result.speech_duration_seconds;
                    const chunks = [];
                    const count = Math.ceil(dur / 25.0);
                    for (let i = 0; i < count; i++) {
                        const start = i * 25.0;
                        if (start >= dur) break;
                        const chunk_num = i + 1;
                        const end = Math.min(dur, start + 30.0);
                        chunks.push({
                            chunk_num,
                            start_time: start,
                            end_time: end,
                            filtered_path_rel: result.filtered_full_path_rel ? `output/filtered_chunk_${chunk_num}_${result.timestamp}.wav` : null,
                            denoised_path_rel: result.denoised_full_path_rel ? `output/denoised_chunk_${chunk_num}_${result.timestamp}.wav` : null
                        });
                    }
                    result.intermediate_chunks = chunks;
                }
                
                setDataset(result);
                setPlayhead(0);
                if (result.intermediate_chunks && result.intermediate_chunks.length > 0) {
                    setSelectedChunk(result.intermediate_chunks[0].chunk_num);
                }
                if (result.id) {
                    setLoadedHistoryId(result.id);
                }
                setStatusMessage('Transcription completed successfully.');
                fetchHistory(1);
            } else {
                throw new Error("Invalid response format received from transcription API.");
            }
        } catch (err) {
            console.error(err);
            setTranscribeError(err.message);
            setStatusMessage('');
        } finally {
            setIsTranscribing(false);
        }
    };

    // ---- draw a lightweight synthetic waveform + segment markers ----
    useEffect(() => {
        const cvs = canvasRef.current;
        if (!cvs) return;
        const dpr = window.devicePixelRatio || 1;
        const w = cvs.clientWidth, h = cvs.clientHeight;
        cvs.width = w * dpr; cvs.height = h * dpr;
        const ctx = cvs.getContext('2d');
        ctx.scale(dpr, dpr);
        ctx.clearRect(0, 0, w, h);

        // deterministic pseudo-waveform bars
        const bars = 260;
        ctx.fillStyle = '#1c2530';
        for (let i = 0; i < bars; i++) {
            const seed = Math.sin(i * 12.9898) * 43758.5453;
            const amp = (Math.abs(seed - Math.floor(seed)) * 0.75 + 0.15);
            const bh = amp * h * 0.8;
            const x = (i / bars) * w;
            ctx.fillRect(x, (h - bh) / 2, w / bars - 1, bh);
        }
        // played portion overlay
        const playedW = (playhead / totalDuration) * w;
        ctx.fillStyle = '#00ff9c';
        ctx.globalAlpha = 0.85;
        ctx.fillRect(0, 0, playedW, h);
        ctx.globalCompositeOperation = 'source-atop';
        for (let i = 0; i < bars; i++) {
            const seed = Math.sin(i * 12.9898) * 43758.5453;
            const amp = (Math.abs(seed - Math.floor(seed)) * 0.75 + 0.15);
            const bh = amp * h * 0.8;
            const x = (i / bars) * w;
            ctx.fillRect(x, (h - bh) / 2, w / bars - 1, bh);
        }
        ctx.globalCompositeOperation = 'source-over';
        ctx.globalAlpha = 1;

        // playhead line
        ctx.strokeStyle = '#ffb000';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(playedW, 0); ctx.lineTo(playedW, h);
        ctx.stroke();
    }, [playhead, totalDuration]);

    const handleWaveClick = (e) => {
        const rect = canvasRef.current.getBoundingClientRect();
        const frac = (e.clientX - rect.left) / rect.width;
        seekTo(Math.max(0, Math.min(totalDuration, frac * totalDuration)));
    };

    // ---- chunk audio playback ----
    const playChunk = () => {
        if (isPlaying) {
            audioElRef.current.pause();
            setIsPlaying(false);
        }

        const ctx = getCtx();
        if (ctx.state === 'suspended') ctx.resume();
        if (chunkSourceRef.current) {
            try { chunkSourceRef.current.stop(); } catch (e) { }
        }
        
        const chunk = data.intermediate_chunks.find((c) => c.chunk_num === selectedChunk);
        if (!chunk) return;

        const relPath = chunkVariant === 'denoised' ? chunk.denoised_path_rel : chunk.filtered_path_rel;
        const url = relPath ? `/${relPath}` : null;

        if (url) {
            setChunkLoading(true);
            fetch(url)
                .then(res => {
                    if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
                    return res.arrayBuffer();
                })
                .then(arrayBuffer => ctx.decodeAudioData(arrayBuffer))
                .then(decodedBuffer => {
                    const currentChunk = data.intermediate_chunks.find((c) => c.chunk_num === selectedChunk);
                    if (!currentChunk || currentChunk.chunk_num !== chunk.chunk_num) {
                        setChunkLoading(false);
                        return;
                    }
                    
                    if (chunkSourceRef.current) {
                        try { chunkSourceRef.current.stop(); } catch (e) { }
                    }
                    const src = ctx.createBufferSource();
                    src.buffer = decodedBuffer;
                    src.connect(ctx.destination);
                    src.onended = () => setChunkPlaying(false);
                    src.start();
                    chunkSourceRef.current = src;
                    setChunkPlaying(true);
                    setChunkLoading(false);
                })
                .catch(err => {
                    console.error("Error playing real chunk audio, falling back to synthetic tone:", err);
                    setChunkLoading(false);
                    const dur = chunk.end_time - chunk.start_time;
                    const buf = generateChunkBuffer(ctx, chunk.chunk_num, chunkVariant, dur);
                    const src = ctx.createBufferSource();
                    src.buffer = buf;
                    src.connect(ctx.destination);
                    src.onended = () => setChunkPlaying(false);
                    src.start();
                    chunkSourceRef.current = src;
                    setChunkPlaying(true);
                });
        } else {
            const dur = chunk.end_time - chunk.start_time;
            const buf = generateChunkBuffer(ctx, chunk.chunk_num, chunkVariant, dur);
            const src = ctx.createBufferSource();
            src.buffer = buf;
            src.connect(ctx.destination);
            src.onended = () => setChunkPlaying(false);
            src.start();
            chunkSourceRef.current = src;
            setChunkPlaying(true);
        }
    };

    const stopChunk = () => {
        if (chunkSourceRef.current) {
            try { chunkSourceRef.current.stop(); } catch (e) { }
        }
        setChunkPlaying(false);
    };

    useEffect(() => stopChunk, [selectedChunk, chunkVariant]); // eslint-disable-line

    const selectedChunkData = data.intermediate_chunks.find((c) => c.chunk_num === selectedChunk);

    return (
        <div className="atc-root" style={styles.root}>
            <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap');
        * { box-sizing: border-box; }
        .tok-verb { color: #ffb000; font-weight: 600; }
        .tok-runway { color: #00ff9c; font-weight: 700; background: rgba(0,255,156,0.08); border-radius: 3px; padding: 0 2px; }
        .tok-alt { color: #6fd3ff; font-weight: 600; }
        .tok-heading { color: #ff8a65; font-weight: 600; }
        .tok-nav { color: #d9a8ff; font-weight: 600; }
        .seg-row:hover { background: #131a22 !important; }
        .btn { cursor: pointer; }
        .submit-btn:hover { background: #00e58b !important; box-shadow: 0 0 18px rgba(0,255,156,0.3) !important; transform: translateY(-1px); }
        .submit-btn:active { background: #00cc7c !important; transform: translateY(0); }
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: #0a0e12; }
        ::-webkit-scrollbar-thumb { background: #2a3644; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #3a4a5a; }

        /* Panel hover glow + pop */
        .atc-panel {
            transition: border-color 0.25s ease, box-shadow 0.25s ease, transform 0.2s ease;
        }
        .atc-panel:hover {
            border-color: #1f3040;
            box-shadow: 0 0 20px rgba(0,255,156,0.04), 0 4px 16px rgba(0,0,0,0.3);
            transform: translateY(-1px);
        }

        /* Stat chip hover */
        .stat-chip {
            transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.15s ease;
        }
        .stat-chip:hover {
            border-color: #00ff9c44;
            box-shadow: 0 0 12px rgba(0,255,156,0.08);
            transform: translateY(-1px) scale(1.02);
        }

        /* Mini stat hover */
        .mini-stat {
            transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.15s ease;
        }
        .mini-stat:hover {
            border-color: #ffb00030;
            box-shadow: 0 0 14px rgba(255,176,0,0.06);
            transform: translateY(-1px);
        }

        /* Play button hover */
        .btn:hover {
            border-color: #3a4a5a !important;
            box-shadow: 0 0 10px rgba(0,255,156,0.06);
        }

        /* Select hover */
        select:hover {
            border-color: #3a4a5a !important;
        }
        select:focus {
            border-color: #00ff9c44 !important;
            outline: none;
            box-shadow: 0 0 8px rgba(0,255,156,0.08);
        }

        /* Segment row active glow */
        .seg-row.seg-active {
            background: rgba(0,255,156,0.04) !important;
            border-left: 2px solid #00ff9c;
        }

        /* History row states */
        .history-row:hover {
            background: #131a22 !important;
            border-color: #2a3644 !important;
        }
        .history-row.history-active {
            background: rgba(0,255,156,0.03) !important;
            border-color: #00ff9c !important;
        }

        /* Active segment HUD subtle pulse */
        @keyframes hudPulse {
            0%, 100% { border-color: #1a232d; }
            50% { border-color: #1f3040; }
        }
        .hud-active {
            animation: hudPulse 3s ease-in-out infinite;
        }

        /* Logo mark hover */
        .logo-mark {
            transition: box-shadow 0.25s ease, transform 0.2s ease;
        }
        .logo-mark:hover {
            box-shadow: 0 0 16px rgba(0,255,156,0.15);
            transform: scale(1.05);
        }

        /* Background ATC watermark */
        .atc-root::before {
            content: 'ATC';
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-family: 'JetBrains Mono', monospace;
            font-size: 28vw;
            font-weight: 900;
            color: rgba(255,255,255,0.012);
            pointer-events: none;
            z-index: 0;
            letter-spacing: 0.05em;
            user-select: none;
        }
        .atc-root > * {
            position: relative;
            z-index: 1;
        }

        /* Classification banner */
        .classification-bar {
            text-align: center;
            font-family: 'JetBrains Mono', monospace;
            font-size: 9px;
            letter-spacing: 3px;
            text-transform: uppercase;
            color: #ffb000;
            padding: 5px 0;
            border-bottom: 1px solid #1a232d;
            opacity: 0.7;
        }

        /* Header designation code blink */
        @keyframes opBlink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
        .op-status-dot {
            display: inline-block;
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #00ff9c;
            animation: opBlink 2s ease-in-out infinite;
            margin-right: 6px;
            vertical-align: middle;
        }
      `}</style>

            {/* Classification Banner */}
            <div className="classification-bar">Ground Station Operations — Authorised Personnel Only</div>

            {/* Header */}
            <div style={styles.header}>
                <div style={styles.headerLeft}>
                    <div className="logo-mark" style={styles.logoMark}>
                        <div style={{ fontSize: 8, letterSpacing: 2, color: '#5f6f80', marginBottom: 2 }}>SYS</div>
                        <div>ATC</div>
                    </div>
                    <div>
                        <div style={styles.title}>
                            <span style={{ color: '#5f6f80', fontWeight: 400, fontSize: 13, fontFamily: mono, letterSpacing: 1.5, display: 'block', marginBottom: 2 }}>ASR-FOR-ATC</span>
                            Air Traffic Control <span style={{ color: '#00ff9c' }}>Intelligence</span> Console
                        </div>
                        <div style={styles.subtitle}>
                            <span className="op-status-dot"></span>
                            operational · real-time voice comms transcription & analysis
                        </div>
                    </div>
                </div>
                <div style={styles.headerStats}>
                    <StatChip label="device" value={(data.device || 'cuda').toUpperCase()} accent={data.device !== 'cpu'} />
                    <StatChip label="lang" value={data.language.toUpperCase()} />
                    <StatChip label="vad" value={data.vad_enabled ? 'on' : 'off'} accent={data.vad_enabled} />
                    <StatChip label="noise removal" value={data.noise_removal_enabled ? 'on' : 'off'} accent={data.noise_removal_enabled} />
                    <StatChip label="speedup" value={`${speedup}x`} accent />
                </div>
            </div>

            {/* Processing stats bar */}
            <div style={styles.statsBar}>
                <MiniStat label="speech duration" value={fmtTime(data.speech_duration_seconds)} />
                <MiniStat label="processing time" value={`${data.processing_time_seconds.toFixed(2)}s`} />
                <MiniStat label="segments" value={data.segments ? data.segments.length : 0} />
                <MiniStat label="run timestamp" value={data.timestamp} />
            </div>

            {/* Top Grid: Transcribe Panel + History Panel */}
            <div style={styles.topGrid}>
                {/* Transcribe Panel */}
                <div className="atc-panel" style={styles.panel}>
                    <div style={styles.panelHeadRow}>
                        <span style={styles.panelLabel}>Transcribe New Audio File</span>
                        {isTranscribing && <span style={styles.loaderSpinner}>Processing...</span>}
                    </div>
                    <form onSubmit={handleTranscribe} style={styles.formRow}>
                        <div style={styles.formField}>
                            <label style={styles.fieldLabel}>Audio File</label>
                            <input
                                type="file"
                                accept="audio/*"
                                onChange={(e) => setFile(e.target.files[0])}
                                style={styles.fileInput}
                                required
                            />
                        </div>
                        <div style={{ ...styles.formField, height: 50, justifyContent: 'flex-end', paddingBottom: 8 }}>
                            <span style={styles.plainLangText}>Language = English</span>
                        </div>
                        <div style={styles.formField}>
                            <label style={styles.fieldLabel}>Compute Device</label>
                            <select
                                value={transcribeDevice}
                                onChange={(e) => setTranscribeDevice(e.target.value)}
                                style={styles.select}
                            >
                                <option value="cuda">CUDA (GPU)</option>
                                <option value="cpu">CPU (Lightweight)</option>
                            </select>
                        </div>
                        <div style={styles.formField}>
                            <label style={styles.fieldLabel}>Noise Removal</label>
                            <div style={styles.checkboxContainer}>
                                <input
                                    type="checkbox"
                                    checked={applyNoiseRemoval}
                                    onChange={(e) => setApplyNoiseRemoval(e.target.checked)}
                                    style={styles.checkbox}
                                    id="applyNoiseRemoval"
                                />
                                <label htmlFor="applyNoiseRemoval" style={styles.checkboxLabel}>Enable</label>
                            </div>
                        </div>
                        <div style={styles.formField}>
                            <label style={styles.fieldLabel}>Noise Red. Coeff.</label>
                            <input
                                type="number"
                                min="0.0"
                                max="1.0"
                                step="0.05"
                                value={noiseReduction}
                                onChange={(e) => setNoiseReduction(parseFloat(e.target.value))}
                                style={styles.numberInput}
                                disabled={!applyNoiseRemoval}
                            />
                        </div>
                        <div style={styles.formField}>
                            <label style={styles.fieldLabel}>Save Chunks</label>
                            <div style={styles.checkboxContainer}>
                                <input
                                    type="checkbox"
                                    checked={saveIntermediate}
                                    onChange={(e) => setSaveIntermediate(e.target.checked)}
                                    style={styles.checkbox}
                                    id="saveIntermediate"
                                />
                                <label htmlFor="saveIntermediate" style={styles.checkboxLabel}>Enable</label>
                            </div>
                        </div>
                        <button
                            type="submit"
                            className="btn submit-btn"
                            disabled={isTranscribing || !file}
                            style={{
                                ...styles.submitBtn,
                                opacity: (isTranscribing || !file) ? 0.6 : 1,
                                cursor: (isTranscribing || !file) ? 'not-allowed' : 'pointer'
                            }}
                        >
                            {isTranscribing ? 'Transcribing...' : 'Upload & Transcribe'}
                        </button>
                    </form>
                    {statusMessage && <div style={styles.statusMsg}>{statusMessage}</div>}
                    {transcribeError && <div style={styles.errorMsg}>Error: {transcribeError}</div>}
                </div>

                {/* Transcription History Panel */}
                <div className="atc-panel" style={{ ...styles.panel, display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
                    <div>
                        <div style={styles.panelHeadRow}>
                            <span style={styles.panelLabel}>Transcription History</span>
                            {historyLoading && <span style={styles.loaderSpinner}>Loading...</span>}
                        </div>
                        
                        <div style={styles.historyList}>
                            {historyList.length === 0 ? (
                                <div style={styles.historyEmpty}>No previous transcriptions found.</div>
                            ) : (
                                historyList.map((item) => (
                                    <div
                                        key={item.id}
                                        onClick={() => loadHistoryItem(item.id)}
                                        className={`history-row ${item.id === loadedHistoryId ? 'history-active' : ''}`}
                                    >
                                        <div style={styles.historyRowTop}>
                                            <span style={styles.historyFilename} title={item.filename}>{item.filename}</span>
                                            <span style={styles.historyTime}>{fmtTime(item.speech_duration_seconds)}</span>
                                        </div>
                                        <div style={styles.historyRowBottom}>
                                            <span style={styles.historyTimestamp}>
                                                {item.created_at ? item.created_at.split(' ')[0] : (item.timestamp || 'unknown')}
                                            </span>
                                            <span style={styles.historyMeta}>
                                                device: <span style={{ color: item.device === 'cpu' ? '#ffb000' : '#00ff9c' }}>{item.device}</span>
                                            </span>
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>

                    {/* Pagination Controls */}
                    {historyTotalPages > 1 && (
                        <div style={styles.historyPagination}>
                            <button
                                type="button"
                                className="btn"
                                onClick={() => fetchHistory(historyPage - 1)}
                                disabled={historyPage <= 1}
                                style={{ ...styles.paginationBtn, opacity: historyPage <= 1 ? 0.4 : 1 }}
                            >
                                ◀ Prev
                            </button>
                            <span style={styles.paginationText}>
                                {historyPage} / {historyTotalPages}
                            </span>
                            <button
                                type="button"
                                className="btn"
                                onClick={() => fetchHistory(historyPage + 1)}
                                disabled={historyPage >= historyTotalPages}
                                style={{ ...styles.paginationBtn, opacity: historyPage >= historyTotalPages ? 0.4 : 1 }}
                            >
                                Next ▶
                            </button>
                        </div>
                    )}
                </div>
            </div>

            {/* Waveform + transport */}
            <div className="atc-panel" style={styles.panel}>
                <div style={styles.panelHeadRow}>
                    <span style={styles.panelLabel}>audio timeline</span>
                    <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
                        <div style={styles.toggleGroup}>
                            <button
                                className="btn"
                                onClick={() => setFullVariant('filtered')}
                                style={{ ...styles.toggleBtn, ...(fullVariant === 'filtered' ? styles.toggleBtnActive : {}) }}
                            >filtered</button>
                            <button
                                className="btn"
                                onClick={() => setFullVariant('denoised')}
                                style={{ ...styles.toggleBtn, ...(fullVariant === 'denoised' ? styles.toggleBtnActive : {}) }}
                            >denoised</button>
                        </div>
                        <span style={styles.timecode}>{fmtTime(playhead)} / {fmtTime(totalDuration)}</span>
                        <button className="btn" onClick={togglePlay} style={styles.playBtn}>
                            {isPlaying ? '⏸ pause' : '▶ play'}
                        </button>
                    </div>
                </div>
                <canvas
                    ref={canvasRef}
                    onClick={handleWaveClick}
                    style={styles.waveCanvas}
                />

                {/* Active Segment HUD / Subtitles */}
                <div className={activeSegIdx >= 0 ? 'hud-active' : ''} style={styles.activeSegmentHud}>
                    {activeSegIdx >= 0 && data.segments && data.segments[activeSegIdx] ? (
                        <div style={styles.activeSegmentContent}>
                            <span style={styles.hudTimecode}>
                                [{fmtTime(data.segments[activeSegIdx].start)} - {fmtTime(data.segments[activeSegIdx].end)}]
                            </span>
                            <span style={styles.hudText}>
                                {tagText(data.segments[activeSegIdx].text).map((n, i) => n.cls
                                    ? <span key={i} className={n.cls}>{n.t}</span>
                                    : <span key={i}>{n.t}</span>)}
                            </span>
                        </div>
                    ) : (
                        <div style={styles.hudPlaceholder}>[ No active transmission at current timecode ]</div>
                    )}
                </div>
            </div>

            {/* Two-column: transcript + chunk pipeline */}
            <div style={styles.mainGrid}>
                {/* Transcript */}
                <div className="atc-panel" style={styles.panel}>
                    <div style={styles.panelHeadRow}>
                        <span style={styles.panelLabel}>transcript</span>
                        <div style={styles.toggleGroup}>
                            <button
                                className="btn"
                                onClick={() => setViewMode('segments')}
                                style={{ ...styles.toggleBtn, ...(viewMode === 'segments' ? styles.toggleBtnActive : {}) }}
                            >segments</button>
                            <button
                                className="btn"
                                onClick={() => setViewMode('full')}
                                style={{ ...styles.toggleBtn, ...(viewMode === 'full' ? styles.toggleBtnActive : {}) }}
                            >full text</button>
                        </div>
                    </div>

                    {viewMode === 'segments' ? (
                        <div ref={segListRef} style={styles.segList}>
                            {data.segments.map((seg, idx) => (
                                <div
                                    key={idx}
                                    ref={(el) => (segRefs.current[idx] = el)}
                                    className={`seg-row ${idx === activeSegIdx ? 'seg-active' : ''}`}
                                    onClick={() => seekTo(seg.start)}
                                    style={{
                                        ...styles.segRow,
                                        background: idx === activeSegIdx ? 'rgba(0,255,156,0.04)' : 'transparent',
                                    }}
                                >
                                    <span style={styles.segTime}>{fmtTime(seg.start)}</span>
                                    <span style={styles.segText}>
                                        {tagText(seg.text).map((n, i) => n.cls
                                            ? <span key={i} className={n.cls}>{n.t}</span>
                                            : <span key={i}>{n.t}</span>)}
                                    </span>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div style={styles.fullText}>
                            {tagText(data.text).map((n, i) => n.cls
                                ? <span key={i} className={n.cls}>{n.t}</span>
                                : <span key={i}>{n.t}</span>)}
                        </div>
                    )}

                    <div style={styles.legend}>
                        <LegendItem cls="tok-verb" label="clearance / instruction" />
                        <LegendItem cls="tok-runway" label="runway" />
                        <LegendItem cls="tok-alt" label="altitude" />
                        <LegendItem cls="tok-heading" label="heading" />
                        <LegendItem cls="tok-nav" label="nav aid" />
                    </div>
                </div>

                {/* Chunk pipeline */}
                <div className="atc-panel" style={styles.panel}>
                    <div style={styles.panelHeadRow}>
                        <span style={styles.panelLabel}>intermediate audio pipeline</span>
                        <span style={styles.chunkCount}>{data.intermediate_chunks.length} chunks</span>
                    </div>

                    <div style={styles.chunkControls}>
                        <label style={styles.fieldLabel}>chunk</label>
                        <select
                            value={selectedChunk}
                            onChange={(e) => setSelectedChunk(Number(e.target.value))}
                            style={styles.select}
                        >
                            {data.intermediate_chunks.map((c) => (
                                <option key={c.chunk_num} value={c.chunk_num}>
                                    chunk {c.chunk_num} · {fmtTime(c.start_time)}–{fmtTime(c.end_time)}
                                </option>
                            ))}
                        </select>

                        <label style={styles.fieldLabel}>variant</label>
                        <select
                            value={chunkVariant}
                            onChange={(e) => setChunkVariant(e.target.value)}
                            style={styles.select}
                        >
                            <option value="filtered">filtered</option>
                            <option value="denoised">denoised</option>
                        </select>

                        <div style={{ display: 'flex', gap: 8, marginTop: 4 }}>
                            <button
                                className="btn"
                                onClick={chunkPlaying ? stopChunk : playChunk}
                                disabled={chunkLoading}
                                style={{ ...styles.playBtn, flex: 1 }}
                            >
                                {chunkLoading ? '⏳ loading...' : chunkPlaying ? '⏹ stop' : '▶ play chunk'}
                            </button>
                        </div>
                    </div>

                    {selectedChunkData && (
                        <div style={styles.chunkMeta}>
                            <div style={styles.chunkMetaRow}>
                                <span style={styles.metaKey}>window</span>
                                <span style={styles.metaVal}>{fmtTime(selectedChunkData.start_time)} → {fmtTime(selectedChunkData.end_time)}</span>
                            </div>
                            <div style={styles.chunkMetaRow}>
                                <span style={styles.metaKey}>span</span>
                                <span style={styles.metaVal}>{(selectedChunkData.end_time - selectedChunkData.start_time).toFixed(0)}s</span>
                            </div>
                            <div style={styles.chunkMetaRow}>
                                <span style={styles.metaKey}>variant</span>
                                <span style={{ ...styles.metaVal, color: chunkVariant === 'denoised' ? '#00ff9c' : '#ffb000' }}>
                                    {chunkVariant}
                                </span>
                            </div>
                        </div>
                    )}

                </div>
            </div>

            <div style={styles.footer}>
                note — this build automatically plays the real audio files (denoised and filtered) served from 
                the backend output directory when available, and falls back to synthesized placeholder audio 
                for preview if the server files are offline.
            </div>
        </div>
    );
}

function StatChip({ label, value, accent }) {
    return (
        <div className="stat-chip" style={{ ...styles.statChip, borderColor: accent ? '#00ff9c44' : '#2a3644' }}>
            <span style={styles.statChipLabel}>{label}</span>
            <span style={{ ...styles.statChipValue, color: accent ? '#00ff9c' : '#e8edf2' }}>{value}</span>
        </div>
    );
}

function MiniStat({ label, value }) {
    return (
        <div className="mini-stat" style={styles.miniStat}>
            <div style={styles.miniStatLabel}>{label}</div>
            <div style={styles.miniStatValue}>{value}</div>
        </div>
    );
}

function LegendItem({ cls, label }) {
    return (
        <div style={styles.legendItem}>
            <span className={cls} style={{ padding: '1px 6px', borderRadius: 3 }}>abc</span>
            <span style={styles.legendLabel}>{label}</span>
        </div>
    );
}

const mono = "'JetBrains Mono', monospace";
const sans = "'Inter', sans-serif";

const styles = {
    root: {
        background: '#0a0e12',
        backgroundImage: 'radial-gradient(ellipse 80% 50% at 15% 10%, rgba(0,255,156,0.015) 0%, transparent 60%), radial-gradient(ellipse 60% 40% at 85% 80%, rgba(255,176,0,0.01) 0%, transparent 50%)',
        color: '#e8edf2',
        fontFamily: sans,
        minHeight: '100vh',
        padding: 20,
        display: 'flex',
        flexDirection: 'column',
        gap: 16,
        position: 'relative',
        overflow: 'hidden',
    },
    header: {
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: 12,
        paddingBottom: 16,
        borderBottom: '1px solid #1a232d',
    },
    headerLeft: { display: 'flex', alignItems: 'center', gap: 14 },
    logoMark: {
        fontFamily: mono, fontWeight: 700, fontSize: 16,
        background: '#0f1620', border: '1px solid #00ff9c55', color: '#00ff9c',
        padding: '8px 12px', borderRadius: 4, letterSpacing: 2,
        textAlign: 'center', lineHeight: 1.1,
    },
    title: { fontSize: 17, fontWeight: 700, letterSpacing: 0.3, lineHeight: 1.4 },
    subtitle: { fontSize: 11, color: '#5f6f80', fontFamily: mono, marginTop: 3, letterSpacing: 0.3 },
    headerStats: { display: 'flex', gap: 8, flexWrap: 'wrap' },
    statChip: {
        display: 'flex', flexDirection: 'column', gap: 2,
        border: '1px solid #2a3644', borderRadius: 5, padding: '5px 10px',
        background: '#0f1620', minWidth: 68,
    },
    statChipLabel: { fontSize: 9.5, color: '#5f6f80', fontFamily: mono, textTransform: 'uppercase', letterSpacing: 0.5 },
    statChipValue: { fontSize: 13, fontFamily: mono, fontWeight: 700 },

    statsBar: {
        display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10,
    },
    miniStat: {
        background: '#0f1620', border: '1px solid #1a232d', borderRadius: 6, padding: '10px 14px',
    },
    miniStatLabel: { fontSize: 10.5, color: '#5f6f80', fontFamily: mono, textTransform: 'uppercase', letterSpacing: 0.6, marginBottom: 4 },
    miniStatValue: { fontSize: 16, fontFamily: mono, fontWeight: 700, color: '#ffb000' },

    panel: {
        background: '#0d131a', border: '1px solid #1a232d', borderRadius: 8, padding: 14,
    },
    panelHeadRow: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 },
    panelLabel: { fontSize: 11, color: '#7d8b9a', fontFamily: mono, textTransform: 'uppercase', letterSpacing: 0.8, fontWeight: 700 },
    timecode: { fontFamily: mono, fontSize: 12, color: '#00ff9c' },
    playBtn: {
        background: '#131b24', border: '1px solid #2a3644', color: '#e8edf2',
        borderRadius: 5, padding: '6px 12px', fontFamily: mono, fontSize: 12, fontWeight: 600,
    },
    waveCanvas: { width: '100%', height: 90, display: 'block', borderRadius: 4, cursor: 'pointer' },

    mainGrid: {
        display: 'grid', gridTemplateColumns: '1.4fr 1fr', gap: 16,
    },

    toggleGroup: { display: 'flex', gap: 4, background: '#0f1620', border: '1px solid #1a232d', borderRadius: 5, padding: 2 },
    toggleBtn: { background: 'transparent', border: 'none', color: '#6b7a8a', fontFamily: mono, fontSize: 11, padding: '4px 10px', borderRadius: 4 },
    toggleBtnActive: { background: '#1a232d', color: '#00ff9c' },

    segList: { maxHeight: 380, overflowY: 'auto', display: 'flex', flexDirection: 'column', position: 'relative' },
    segRow: { display: 'flex', gap: 12, padding: '7px 10px', cursor: 'pointer', alignItems: 'baseline' },
    segTime: { fontFamily: mono, fontSize: 11, color: '#5f6f80', minWidth: 58, flexShrink: 0 },
    segText: { fontSize: 13, lineHeight: 1.6, color: '#c3ccd4' },

    fullText: { fontSize: 13, lineHeight: 1.9, color: '#c3ccd4', maxHeight: 380, overflowY: 'auto', padding: '4px 4px' },

    legend: { display: 'flex', gap: 14, flexWrap: 'wrap', marginTop: 12, paddingTop: 10, borderTop: '1px solid #1a232d' },
    legendItem: { display: 'flex', alignItems: 'center', gap: 6 },
    legendLabel: { fontSize: 10.5, color: '#5f6f80', fontFamily: mono },

    chunkCount: { fontFamily: mono, fontSize: 11, color: '#5f6f80' },
    chunkControls: { display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 12 },
    fieldLabel: { fontSize: 10, color: '#5f6f80', fontFamily: mono, textTransform: 'uppercase', letterSpacing: 0.6, marginTop: 4 },
    select: {
        background: '#0f1620', border: '1px solid #2a3644', color: '#e8edf2',
        borderRadius: 5, padding: '7px 10px', fontFamily: mono, fontSize: 12,
    },

    chunkMeta: { background: '#0f1620', border: '1px solid #1a232d', borderRadius: 6, padding: 10, marginBottom: 14 },
    chunkMetaRow: { display: 'flex', justifyContent: 'space-between', padding: '3px 0' },
    metaKey: { fontSize: 11, color: '#5f6f80', fontFamily: mono },
    metaVal: { fontSize: 11, color: '#e8edf2', fontFamily: mono, fontWeight: 700 },

    chunkStrip: { position: 'relative', height: 30, background: '#0f1620', borderRadius: 4, border: '1px solid #1a232d' },
    stripCaption: { fontSize: 10, color: '#4a5868', fontFamily: mono, marginTop: 6, textAlign: 'center' },

    footer: {
        fontSize: 11, color: '#4a5868', fontFamily: mono, lineHeight: 1.6,
        borderTop: '1px solid #1a232d', paddingTop: 12,
    },
    formRow: {
        display: 'flex',
        flexWrap: 'wrap',
        gap: 12,
        alignItems: 'flex-end',
        marginTop: 6,
    },
    formField: {
        display: 'flex',
        flexDirection: 'column',
        gap: 4,
        flex: '1 1 180px',
    },
    fileInput: {
        background: '#0f1620',
        border: '1px solid #2a3644',
        color: '#e8edf2',
        borderRadius: 5,
        padding: '6px 10px',
        fontFamily: mono,
        fontSize: 12,
        cursor: 'pointer',
    },
    checkboxContainer: {
        display: 'flex',
        alignItems: 'center',
        gap: 6,
        height: 32,
    },
    checkbox: {
        cursor: 'pointer',
        accentColor: '#00ff9c',
        width: 16,
        height: 16,
    },
    checkboxLabel: {
        fontSize: 12,
        color: '#c3ccd4',
        cursor: 'pointer',
        userSelect: 'none',
    },
    numberInput: {
        background: '#0f1620',
        border: '1px solid #2a3644',
        color: '#e8edf2',
        borderRadius: 5,
        padding: '7px 10px',
        fontFamily: mono,
        fontSize: 12,
        width: '100%',
    },
    submitBtn: {
        background: '#00ff9c',
        border: 'none',
        color: '#0a0e12',
        borderRadius: 5,
        padding: '8px 16px',
        fontFamily: mono,
        fontSize: 12,
        fontWeight: 700,
        height: 32,
        alignSelf: 'flex-end',
        transition: 'background 0.2s ease',
    },
    statusMsg: {
        fontSize: 12,
        color: '#00ff9c',
        fontFamily: mono,
        marginTop: 8,
    },
    errorMsg: {
        fontSize: 12,
        color: '#ff8a65',
        fontFamily: mono,
        marginTop: 8,
    },
    loaderSpinner: {
        fontSize: 11,
        color: '#ffb000',
        fontFamily: mono,
        textTransform: 'uppercase',
    },
    activeSegmentHud: {
        background: '#070a0e',
        border: '1px solid #1a232d',
        borderRadius: 6,
        padding: '12px 16px',
        marginTop: 12,
        minHeight: 56,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
    },
    activeSegmentContent: {
        display: 'flex',
        gap: 12,
        alignItems: 'center',
        width: '100%',
    },
    hudTimecode: {
        fontFamily: mono,
        fontSize: 12,
        color: '#ffb000',
        whiteSpace: 'nowrap',
    },
    hudText: {
        fontSize: 15,
        fontWeight: 500,
        color: '#e8edf2',
        lineHeight: 1.4,
    },
    hudPlaceholder: {
        fontSize: 12,
        color: '#4a5868',
        fontFamily: mono,
        fontStyle: 'italic',
    },
    plainLangText: {
        fontFamily: mono,
        fontSize: 12,
        color: '#ffb000',
        fontWeight: 600,
        height: 16,
    },
    topGrid: {
        display: 'grid',
        gridTemplateColumns: '1.4fr 1fr',
        gap: 16,
    },
    historyList: {
        maxHeight: 180,
        overflowY: 'auto',
        display: 'flex',
        flexDirection: 'column',
        gap: 4,
        marginTop: 6,
        paddingRight: 4,
    },
    historyEmpty: {
        fontFamily: mono,
        fontSize: 12,
        color: '#5f6f80',
        padding: '16px 0',
        textAlign: 'center',
        fontStyle: 'italic',
    },
    historyRow: {
        display: 'flex',
        flexDirection: 'column',
        padding: '6px 10px',
        background: '#0f1620',
        border: '1px solid #1a232d',
        borderRadius: 4,
        cursor: 'pointer',
        transition: 'all 0.2s ease',
    },
    historyRowTop: {
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 2,
    },
    historyFilename: {
        fontSize: 12.5,
        fontWeight: 600,
        color: '#e8edf2',
        whiteSpace: 'nowrap',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        maxWidth: '80%',
    },
    historyTime: {
        fontFamily: mono,
        fontSize: 11,
        color: '#ffb000',
    },
    historyRowBottom: {
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
    },
    historyTimestamp: {
        fontFamily: mono,
        fontSize: 10,
        color: '#5f6f80',
    },
    historyMeta: {
        fontFamily: mono,
        fontSize: 10,
        color: '#5f6f80',
    },
    historyPagination: {
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        gap: 12,
        marginTop: 8,
        borderTop: '1px solid #1a232d',
        paddingTop: 8,
    },
    paginationBtn: {
        background: '#131b24',
        border: '1px solid #2a3644',
        color: '#e8edf2',
        borderRadius: 4,
        padding: '4px 10px',
        fontFamily: mono,
        fontSize: 11,
        cursor: 'pointer',
    },
    paginationText: {
        fontFamily: mono,
        fontSize: 11.5,
        color: '#5f6f80',
    },
};