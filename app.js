/**
 * CineMatch Static Site — Client-Side Application Logic
 * Features: Tab switching, Autocomplete search, TMDB API, Feature breakdown,
 *           3D Manifold, Dark/Light mode, Vibe analysis
 */

// ============ CONFIGURATION ============
const TMDB_API_KEYS = [
    '8265bd1679663a7ea12ac168da84d2e8',
    'e99b89ffb15842de8b76fc35ae80955e'
];
const POSTER_BASE_URL = 'https://image.tmdb.org/t/p/w500/';
const PLACEHOLDER_POSTER = 'https://placehold.co/500x750/0e1117/ffffff.png?text=NO+IMAGE';

function getRandomKey() {
    return TMDB_API_KEYS[Math.floor(Math.random() * TMDB_API_KEYS.length)];
}

// ============ LOGGING CONTROL ============
// Set to true during development to enable console output
const DEBUG = true;
const log = DEBUG ? console : { warn: () => { }, error: () => { } };

// ============ GLOBAL STATE ============
let moviesData = [];
let recommendationsData = {};
let manifoldData = [];
let dataLoaded = false;
let currentQueryTitle = '';

// ============ INITIALIZATION ============
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initThemeToggle();
    initSearch();
    loadAllData();
    initSmartSidebar();
});

// ============ DARK / LIGHT MODE ============
function initThemeToggle() {
    const saved = localStorage.getItem('cinematch-theme') || 'dark';
    document.body.setAttribute('data-theme', saved);

    const toggle = document.getElementById('theme-toggle');
    if (toggle) {
        toggle.textContent = saved === 'dark' ? 'LIGHT MODE' : 'DARK MODE';
        toggle.addEventListener('click', () => {
            const current = document.body.getAttribute('data-theme');
            const next = current === 'dark' ? 'light' : 'dark';
            document.body.setAttribute('data-theme', next);
            localStorage.setItem('cinematch-theme', next);
            toggle.textContent = next === 'dark' ? 'LIGHT MODE' : 'DARK MODE';

            // Re-render manifold if it was already rendered to match new theme
            if (window._manifoldRendered) {
                window._manifoldRendered = false;
                renderManifold();
            }
        });
    }
}

// ============ TAB SYSTEM ============
function initTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.dataset.tab;

            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            const panel = document.getElementById(`tab-${tabId}`);
            if (panel) panel.classList.add('active');

            // Lazy-load manifold on first visit
            if (tabId === 'manifold' && manifoldData.length > 0 && !window._manifoldRendered) {
                // Wait for tab panel to be visible, then render
                requestAnimationFrame(() => {
                    requestAnimationFrame(() => {
                        renderManifold();
                    });
                });
            }
        });
    });
}

// ============ DATA LOADING ============
async function loadAllData() {
    try {
        const [moviesRes, recsRes, manifoldRes] = await Promise.all([
            fetch('data/movies.json'),
            fetch('data/recommendations.json'),
            fetch('data/manifold.json')
        ]);

        if (!moviesRes.ok || !recsRes.ok) {
            throw new Error('Failed to load core data files');
        }

        moviesData = await moviesRes.json();
        recommendationsData = await recsRes.json();

        if (manifoldRes.ok) {
            manifoldData = await manifoldRes.json();
        }

        dataLoaded = true;
        loadEvaluationData();
    } catch (err) {
        log.error('Data load error:', err);
        const area = document.getElementById('recommendations-area');
        if (area) {
            area.innerHTML = `
                <div style="text-align:center; padding:50px; font-family:'JetBrains Mono', monospace;">
                    <p style="color:#ff3e3e; font-size:1rem; letter-spacing:1px;">DATA ARTIFACTS NOT FOUND</p>
                    <p style="color:#888; font-size:0.85rem; margin-top:15px;">
                        Run <code style="background:#1a1b26; padding:2px 8px; border-radius:4px; color:#00ff41;">python export_data.py</code> to generate the required JSON data files.
                    </p>
                </div>`;
        }
    }
}

async function loadEvaluationData() {
    try {
        const res = await fetch('data/evaluation.json');
        if (!res.ok) return;
        const data = await res.json();

        data.forEach(item => {
            const method = item.method.toLowerCase();
            const precEl = document.getElementById(`${method}-precision`);
            const latEl = document.getElementById(`${method}-latency`);

            if (precEl) precEl.textContent = item.precision.toFixed(4);
            if (latEl) latEl.textContent = item.latency.toFixed(2) + 'ms';
        });
    } catch (err) {
        log.warn('Failed to load evaluation results:', err.message);
    }
}

// ============ SEARCH / AUTOCOMPLETE ============
function initSearch() {
    const input = document.getElementById('movie-search');
    const dropdown = document.getElementById('autocomplete-dropdown');
    if (!input || !dropdown) return;

    let currentHighlight = -1;
    let filteredItems = [];

    input.addEventListener('input', () => {
        const query = input.value.trim().toLowerCase();
        dropdown.innerHTML = '';
        currentHighlight = -1;

        if (query.length < 1) {
            dropdown.classList.remove('visible');
            return;
        }

        const startsWithMatches = moviesData.filter(m => m.title.toLowerCase().startsWith(query));
        const includesMatches = moviesData.filter(m =>
            !m.title.toLowerCase().startsWith(query) && m.title.toLowerCase().includes(query)
        );
        filteredItems = [...startsWithMatches, ...includesMatches].slice(0, 50);

        if (filteredItems.length === 0) {
            dropdown.classList.remove('visible');
            return;
        }

        filteredItems.forEach((movie) => {
            const div = document.createElement('div');
            div.className = 'autocomplete-item';
            div.textContent = movie.title;
            div.addEventListener('click', () => {
                input.value = movie.title;
                dropdown.classList.remove('visible');
                selectMovie(movie.title);
            });
            dropdown.appendChild(div);
        });

        dropdown.classList.add('visible');
    });

    input.addEventListener('keydown', (e) => {
        const items = dropdown.querySelectorAll('.autocomplete-item');
        if (e.key === 'ArrowDown') {
            e.preventDefault();
            currentHighlight = Math.min(currentHighlight + 1, items.length - 1);
            updateHighlight(items, currentHighlight);
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            currentHighlight = Math.max(currentHighlight - 1, 0);
            updateHighlight(items, currentHighlight);
        } else if (e.key === 'Enter') {
            e.preventDefault();
            if (currentHighlight >= 0 && items[currentHighlight]) {
                items[currentHighlight].click();
            } else if (filteredItems.length > 0) {
                input.value = filteredItems[0].title;
                dropdown.classList.remove('visible');
                selectMovie(filteredItems[0].title);
            }
        } else if (e.key === 'Escape') {
            dropdown.classList.remove('visible');
        }
    });

    document.addEventListener('click', (e) => {
        if (!e.target.closest('.select-wrapper')) {
            dropdown.classList.remove('visible');
        }
    });
}

function updateHighlight(items, idx) {
    items.forEach(item => item.classList.remove('highlighted'));
    if (items[idx]) {
        items[idx].classList.add('highlighted');
        items[idx].scrollIntoView({ block: 'nearest' });
    }
}

// ============ MOVIE SELECTION & RECOMMENDATIONS ============
async function selectMovie(title) {
    const area = document.getElementById('recommendations-area');
    if (!area) return;

    currentQueryTitle = title;
    const t_start = performance.now();

    area.innerHTML = `
        <div class="computing-msg">
            COMPUTING HIGH-DIMENSIONAL TRAJECTORY FOR: ${escapeHtml(title)}...
        </div>`;

    const recs = recommendationsData[title];
    if (!recs || recs.length === 0) {
        area.innerHTML = `
            <div style="text-align:center; padding:30px; color:#ff3e3e; font-family:'JetBrains Mono', monospace;">
                No recommendations found for "${escapeHtml(title)}".
            </div>`;
        return;
    }

    const top6 = recs.slice(0, 6);
    const top = top6[0];
    const rest = top6.slice(1);

    // Fetch all posters in parallel
    const allPosters = await Promise.all(top6.map(r => fetchPoster(r.id, r.title)));

    // Fetch details & wiki for primary target
    const [details, wiki] = await Promise.all([
        fetchMovieDetails(top.id),
        fetchWikiSummary(top.title)
    ]);

    const vibe = analyzeVibe(wiki + (details ? details.overview : ''));
    const t_end = performance.now();
    const latency = t_end - t_start;
    updateLatency(latency);

    // ---- PRIMARY TARGET ----
    let primaryHTML = `
        <div class="tech-header">PRIMARY TARGET LOCK</div>
        <div class="primary-target">
            <div class="primary-poster">
                <div class="movie-card" style="border-color:#00ff41;">
                    <img src="${allPosters[0]}" class="poster-img" alt="${escapeHtml(top.title)}" loading="lazy"
                         onerror="this.src='${PLACEHOLDER_POSTER}'">
                </div>
            </div>
            <div class="primary-details">
                <h1>${escapeHtml(top.title)}</h1>
                <div class="vibe-tag">
                    <span class="sigma-badge" style="border-color:#ffcc00; color:#ffcc00;">VIBE_ANALYSIS</span>
                    <span style="font-family:'JetBrains Mono', monospace; color:#ffcc00;">${vibe}</span>
                </div>`;

    if (details) {
        primaryHTML += `
                <div class="detail-badges">
                    <span class="sigma-badge">RATING</span> <span style="color:#fff; font-family:'JetBrains Mono';">${details.rating}</span> &nbsp;&nbsp;
                    <span class="sigma-badge">RELEASE</span> <span style="color:#fff; font-family:'JetBrains Mono';">${details.release_date}</span> &nbsp;&nbsp;
                    <span class="sigma-badge">RUNTIME</span> <span style="color:#fff; font-family:'JetBrains Mono';">${details.runtime}</span>
                </div>`;

        if (details.trailer) {
            primaryHTML += `
                <div class="trailer-embed">
                    <iframe src="https://www.youtube.com/embed/${details.trailerKey}?rel=0"
                            allowfullscreen title="Trailer for ${escapeHtml(top.title)}"></iframe>
                </div>`;
        }
    }

    // Primary target feature breakdown
    if (top.features && top.features.length > 0) {
        primaryHTML += buildFeatureBlock(top.score, top.features);
    }

    primaryHTML += `
                <div class="wiki-block">${escapeHtml(wiki)}</div>
            </div>
        </div>`;

    // ---- RECOMMENDATIONS GRID ----
    let recsHTML = `
        <div class="tech-header">LATENT CLUSTER PROJECTIONS</div>
        <div class="recs-grid">`;

    rest.forEach((rec, i) => {
        const featureId = `features-${i}`;
        recsHTML += `
            <div class="rec-card">
                <div class="movie-card">
                    <img src="${allPosters[i + 1]}" class="poster-img" alt="${escapeHtml(rec.title)}" loading="lazy"
                         onerror="this.src='${PLACEHOLDER_POSTER}'">
                </div>
                <button class="rec-btn" onclick="navigateToMovie('${escapeAttr(rec.title)}')">
                    ANALYZE &rarr;
                </button>
                <button class="explain-btn" onclick="toggleFeatures('${featureId}')">
                    EXPLAIN
                </button>
                <div class="movie-title-label">${escapeHtml(rec.title)}</div>
                <div class="cosine-display" id="${featureId}">
                    <div class="cosine-score">COSINE SIMILARITY: ${rec.score.toFixed(4)}</div>
                    ${rec.features && rec.features.length > 0 ? `
                    <div class="feature-breakdown" style="display:none;">
                        <div class="feature-header">TOP SHARED FEATURES:</div>
                        ${rec.features.map(f => `
                            <div class="feature-row">
                                <span class="feat-name">${escapeHtml(f.f)}</span>
                                <span class="feat-score">${f.c.toFixed(3)}</span>
                            </div>
                        `).join('')}
                    </div>` : ''}
                </div>
            </div>`;
    });

    recsHTML += `</div>`;
    area.innerHTML = primaryHTML + recsHTML;
}

function buildFeatureBlock(score, features) {
    let html = `
        <div class="cosine-display">
            <div class="cosine-score">COSINE SIMILARITY: ${score.toFixed(4)}</div>
            <div class="feature-breakdown">
                <div class="feature-header">TOP SHARED FEATURES:</div>`;

    features.forEach(f => {
        html += `
            <div class="feature-row">
                <span class="feat-name">${escapeHtml(f.f)}</span>
                <span class="feat-score">${f.c.toFixed(3)}</span>
            </div>`;
    });

    html += `</div></div>`;
    return html;
}

function toggleFeatures(featureId) {
    const container = document.getElementById(featureId);
    if (!container) return;
    const breakdown = container.querySelector('.feature-breakdown');
    if (!breakdown) return;

    if (breakdown.style.display === 'none') {
        breakdown.style.display = 'block';
    } else {
        breakdown.style.display = 'none';
    }
}

function navigateToMovie(title) {
    const input = document.getElementById('movie-search');
    if (input) {
        input.value = title;
    }
    selectMovie(title);
    window.scrollTo({ top: 300, behavior: 'smooth' });
}

// ============ TMDB API ============
async function fetchPoster(movieId, title) {
    const key = getRandomKey();
    try {
        const res = await fetch(`https://api.themoviedb.org/3/movie/${movieId}?api_key=${key}`, {
            signal: AbortSignal.timeout(4000)
        });
        if (res.ok) {
            const data = await res.json();
            if (data.poster_path) {
                return `${POSTER_BASE_URL}${data.poster_path}`;
            }
        }

        const searchRes = await fetch(
            `https://api.themoviedb.org/3/search/movie?api_key=${key}&query=${encodeURIComponent(title)}`,
            { signal: AbortSignal.timeout(4000) }
        );
        if (searchRes.ok) {
            const searchData = await searchRes.json();
            if (searchData.results && searchData.results.length > 0 && searchData.results[0].poster_path) {
                return `${POSTER_BASE_URL}${searchData.results[0].poster_path}`;
            }
        }
    } catch (err) {
        log.warn(`Poster fetch failed for ${title}:`, err.message);
    }
    return PLACEHOLDER_POSTER;
}

async function fetchMovieDetails(movieId) {
    const key = getRandomKey();
    try {
        const res = await fetch(
            `https://api.themoviedb.org/3/movie/${movieId}?api_key=${key}&append_to_response=videos`,
            { signal: AbortSignal.timeout(5000) }
        );
        if (!res.ok) return null;

        const data = await res.json();
        const details = {
            rating: data.vote_average || 'N/A',
            release_date: data.release_date || 'N/A',
            runtime: `${data.runtime || 0} min`,
            overview: data.overview || '',
            trailer: null,
            trailerKey: null
        };

        const videos = (data.videos && data.videos.results) || [];
        for (const video of videos) {
            if (video.type === 'Trailer' && video.site === 'YouTube') {
                details.trailer = `https://www.youtube.com/watch?v=${video.key}`;
                details.trailerKey = video.key;
                break;
            }
        }

        return details;
    } catch (err) {
        log.warn('Movie details fetch failed:', err.message);
        return null;
    }
}

async function fetchWikiSummary(title) {
    try {
        const res = await fetch(
            `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(title)}`,
            { signal: AbortSignal.timeout(4000) }
        );
        if (res.ok) {
            const data = await res.json();
            if (data.extract) {
                return data.extract.length > 800
                    ? data.extract.substring(0, 800) + '...'
                    : data.extract;
            }
        }

        const res2 = await fetch(
            `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(title + ' (film)')}`,
            { signal: AbortSignal.timeout(4000) }
        );
        if (res2.ok) {
            const data2 = await res2.json();
            if (data2.extract) {
                return data2.extract.length > 800
                    ? data2.extract.substring(0, 800) + '...'
                    : data2.extract;
            }
        }
    } catch (err) {
        log.warn('Wiki fetch failed:', err.message);
    }
    return 'No Wikipedia entry found.';
}

// ============ VIBE ANALYSIS ============
function analyzeVibe(text) {
    const positive = ['exciting', 'love', 'hero', 'fun', 'happy', 'adventure', 'great', 'brilliant', 'wonderful', 'joy'];
    const dark = ['death', 'murder', 'scary', 'blood', 'terror', 'dark', 'evil', 'ghost', 'war', 'tragedy'];

    const words = text.toLowerCase().split(/\s+/);
    const posScore = words.filter(w => positive.includes(w)).length;
    const darkScore = words.filter(w => dark.includes(w)).length;

    if (darkScore > posScore) return 'DARK / GRITTY';
    if (posScore > darkScore) return 'UPLIFTING / HEROIC';
    return 'BALANCED / MYSTERIOUS';
}

// ============ LATENCY DISPLAY ============
function updateLatency(ms) {
    const display = document.getElementById('latency-display');
    const metricInf = document.getElementById('metric-inference');
    const formatted = ms.toFixed(2) + 'ms';
    if (display) display.textContent = formatted;
    if (metricInf) metricInf.textContent = formatted;
}

// ============ 3D MANIFOLD (Plotly.js) ============
function renderManifold() {
    if (manifoldData.length === 0) return;

    const chartEl = document.getElementById('manifold-chart');
    const loadingEl = document.getElementById('manifold-loading');

    if (!chartEl) return;

    // Check if Plotly is available
    if (typeof Plotly === 'undefined') {
        log.warn('Plotly not loaded yet, retrying in 500ms...');
        setTimeout(renderManifold, 500);
        return;
    }

    // Ensure container is visible and has dimensions
    const rect = chartEl.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) {
        log.warn('Manifold container not visible yet, retrying in 200ms...');
        setTimeout(renderManifold, 200);
        return;
    }

    const isLight = document.body.getAttribute('data-theme') === 'light';

    // Sample max 600 points for performance
    const sampled = manifoldData.length > 600
        ? [...manifoldData].sort(() => Math.random() - 0.5).slice(0, 600)
        : [...manifoldData];

    const trace = {
        x: sampled.map(d => d.x),
        y: sampled.map(d => d.y),
        z: sampled.map(d => d.z),
        text: sampled.map(d => d.title),
        hoverinfo: 'text',
        mode: 'markers',
        type: 'scatter3d',
        marker: {
            size: 3.5,
            color: sampled.map(d => d.z),
            colorscale: 'Plasma',
            opacity: 0.85,
            line: { width: 0 }
        }
    };

    const gridColor = isLight ? 'rgba(0,0,0,0.1)' : 'rgba(200,200,200,0.3)';
    const tickColor = isLight ? '#555' : '#888';
    const bgColor = isLight ? 'rgba(0,0,0,0.03)' : 'rgba(100,100,100,0.1)';

    const axisStyle = {
        showbackground: true,
        backgroundcolor: bgColor,
        gridcolor: gridColor,
        zerolinecolor: gridColor,
        title: '',
        tickfont: { size: 10, color: tickColor }
    };

    const layout = {
        margin: { l: 0, r: 0, b: 10, t: 0 },
        scene: { xaxis: axisStyle, yaxis: axisStyle, zaxis: axisStyle },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        height: 700
    };

    Plotly.newPlot(chartEl, [trace], layout, {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['toImage', 'sendDataToCloud']
    });

    if (loadingEl) loadingEl.classList.add('hidden');
    window._manifoldRendered = true;
}

// ============ UTILITIES ============
function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function escapeAttr(str) {
    if (!str) return '';
    return str.replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/"/g, '&quot;');
}

// ============ SMART SIDEBAR ANIMATIONS ============
let lastMeasuredSearchLatency = 0;

function updateLatency(ms) {
    lastMeasuredSearchLatency = ms;
    const display = document.getElementById('latency-display');
    if (display) {
        display.textContent = ms.toFixed(2) + 'ms';
        display.style.color = '#ffcc00';
        setTimeout(() => { display.style.color = 'var(--accent-green)'; }, 1000);
    }
}

function initSmartSidebar() {
    const bars = document.querySelectorAll('.progress-fill');

    const display = document.getElementById('latency-display');
    if (display) {
        display.textContent = '0.00ms';
    }


    // Subtle drift in the progress bar to show "active scanning"
    let width = 100;
    setInterval(() => {
        const bar = document.getElementById('integrity-bar');
        if (bar) {
            width = 95 + Math.random() * 5;
            bar.style.width = width + '%';
        }
    }, 5000);


}
