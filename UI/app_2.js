// --- DOM Elements ---
const importBtn = document.getElementById('import-btn');
const exportBtn = document.getElementById('export-btn');
const emptyState = document.getElementById('empty-state');
const mainInterface = document.getElementById('main-interface');

const zoneSection = document.getElementById('zone-section');
const imageGrid = document.getElementById('image-grid');
const zoneTitle = document.getElementById('zone-title');

const miniTimelineContainer = document.getElementById('mini-timeline-container');
const miniTrack = document.getElementById('mini-track');
const scrubberWrapper = document.getElementById('scrubber-wrapper');
const scrubPreview = document.getElementById('scrub-preview');
const previewImg = document.getElementById('preview-img');
const previewInfo = document.getElementById('preview-info');

const sliderMin = document.getElementById('slider-min');
const sliderMax = document.getElementById('slider-max');
const rangeWrapper = document.getElementById('range-wrapper');

const modal = document.getElementById('image-modal');
const modalImg = document.getElementById('modal-img');
const closeModal = document.getElementById('close-modal');

// Nouveaux éléments de la navigation (V2)
const workspaceDisplay = document.getElementById('workspace-display');
const currentFolderDisplay = document.getElementById('current-folder-display');
const btnConfigWorkspace = document.getElementById('btn-config-workspace');
const btnLoadExisting = document.getElementById('btn-load-existing');
const thresholdSlider = document.getElementById('threshold-slider');
const thresholdVal = document.getElementById('threshold-val');
let currentThreshold = 0.50;

// Gestion Modal
closeModal.onclick = () => { modal.style.display = 'none'; };
modal.onclick = (e) => { if (e.target === modal) modal.style.display = 'none'; };

function showModal(url) {
    modalImg.src = url;
    modal.style.display = 'flex';
}

// --- STATE ---
let dataPoints = [];
let chartInstance = null;
let currentSelection = { start: -1, end: -1 };
let gridTimeout = null;

// --- PALETTE PREMIUM ---
const COLOR_CLEAN = '#10b981';     // Emerald 500
const COLOR_POLLUTED = '#ef4444';  // Red 500
const COLOR_POLLUTED_BG = 'rgba(239, 68, 68, 0.2)'; 

// --- INITIALISATION WORKSPACE ---
let currentWorkspace = "";
async function loadConfig() {
    try {
        let res = await fetch('http://127.0.0.1:5000/get_config');
        let data = await res.json();
        if (data.workspace_dir) {
            currentWorkspace = data.workspace_dir;
            workspaceDisplay.value = currentWorkspace;
        }
    } catch(e) { console.warn("Impossible de charger la config", e); }
}
loadConfig();

// --- EVENT LISTENERS ---
btnConfigWorkspace.addEventListener('click', async () => {
    if (!window.pywebview || !window.pywebview.api) { alert("Mode natif requis."); return; }
    let newWorkspace = await window.pywebview.api.open_folder_dialog("Sélectionner le répertoire de travail");
    if (newWorkspace) {
        currentWorkspace = newWorkspace;
        workspaceDisplay.value = newWorkspace;
        fetch('http://127.0.0.1:5000/set_config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ workspace_dir: newWorkspace })
        }).catch(err => console.error(err));
    }
});

btnLoadExisting.addEventListener('click', handleLoadExisting);
importBtn.addEventListener('click', handleImportNative);
exportBtn.addEventListener('click', handleExport);

// --- 1. OUVRIR DOSSIER EXISTANT (SANS IA) ---
async function handleLoadExisting() {
    if (!window.pywebview || !window.pywebview.api) {
        alert("Mode natif non détecté.");
        return;
    }

    if (!currentWorkspace) {
        alert("Veuillez d'abord choisir un répertoire de travail.");
        return;
    }

    let targetFolder = await window.pywebview.api.open_folder_dialog("Sélectionnez un dossier déjà analysé");
    if (!targetFolder) return;

    currentFolderDisplay.value = targetFolder;

    // UI Loading
    emptyState.style.display = 'none';
    mainInterface.style.display = 'flex';
    exportBtn.disabled = true;

    const loadingScreen = document.getElementById('loading-screen');
    const loadingCount = document.getElementById('loading-count');
    if (loadingScreen && loadingCount) {
        loadingScreen.style.display = 'flex';
        loadingCount.textContent = "... Lecture du fichier CSV existant ...";
    }

    try {
        const response = await fetch('http://127.0.0.1:5000/load_existing', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ folder_path: targetFolder })
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.error || "Erreur lors de la lecture.");
        }

        const data = await response.json();
        window.currentDestDir = data.dest_dir; // Important pour la sauvegarde future

        // Formater les données pour le graphique
        dataPoints = data.predictions.map((pred, i) => {
            let cleanPath = pred.path.startsWith('/') ? pred.path.substring(1) : pred.path;

            // NOUVEAU : Si le CSV indique un label différent du score à 0.5, on considère que c'est un override manuel.
            let defaultLabel = pred.score >= currentThreshold ? 1 : 0;
            let isOverridden = pred.label !== defaultLabel;

            return {
                id: i,
                name: pred.name,
                date: pred.date,
                time: pred.time,
                shortDate: pred.date !== "N/A" ? pred.date.split('/')[0] : "N/A",
                shortTime: pred.time !== "N/A" ? pred.time.split(':').slice(0, 2).join(':') : "N/A",
                url: `http://127.0.0.1:5000/image/${cleanPath}`,
                path: pred.path,
                originalScore: pred.score,
                label: pred.label, // Label actuel (venant du CSV ou de l'IA)
                status: pred.status,
                manualOverride: isOverridden // Verrou pour ne pas écraser les choix de l'utilisateur
            };
        });

        if (loadingScreen) loadingScreen.style.display = 'none';
        exportBtn.disabled = false;
        miniTimelineContainer.style.display = 'block';

        if (dataPoints.length > 0) {
            const periodEl = document.getElementById('analysis-period');
            if (periodEl) {
                const first = dataPoints[0];
                const last = dataPoints[dataPoints.length - 1];
                periodEl.textContent = `Période : ${first.date} ${first.shortTime} au ${last.date} ${last.shortTime}`;
            }
            initSliders(dataPoints.length);
            renderMiniTimeline();
            initScrubber();
            renderChart();
        } else {
            alert("Aucune image trouvée dans le CSV.");
        }

    } catch (e) {
        alert("Erreur : " + e.message);
        if (loadingScreen) loadingScreen.style.display = 'none';
        emptyState.style.display = 'flex';
        mainInterface.style.display = 'none';
    }
}

// --- LOGIQUE DU SEUIL IA ---
const lockThresholdBtn = document.getElementById('lock-threshold-btn');

lockThresholdBtn.addEventListener('click', () => {
    if (thresholdSlider.disabled) {
        if (confirm("Attention : toutes les labellisations manuelles risquent d'être modifiées si vous changez le seuil. Voulez-vous déverrouiller le curseur ?")) {
            thresholdSlider.disabled = false;
            thresholdVal.disabled = false;
            lockThresholdBtn.textContent = '🔓 LIBRE';
            lockThresholdBtn.style.color = '';
            lockThresholdBtn.style.borderColor = '';
        }
    } else {
        thresholdSlider.disabled = true;
        thresholdVal.disabled = true;
        lockThresholdBtn.textContent = '🔒 VERROUILLÉ';
        lockThresholdBtn.style.color = '';
        lockThresholdBtn.style.borderColor = '';
    }
});

function applyThresholdChange(newVal) {
    currentThreshold = parseFloat(newVal);
    if (isNaN(currentThreshold)) currentThreshold = 0.5;
    if (currentThreshold < 0) currentThreshold = 0;
    if (currentThreshold > 1) currentThreshold = 1;

    thresholdSlider.value = currentThreshold;
    thresholdVal.value = currentThreshold.toFixed(2);

    if (dataPoints.length > 0) {
        // Mettre à jour les labels qui n'ont pas été forcés manuellement
        dataPoints.forEach(dp => {
            if (!dp.manualOverride) {
                dp.label = dp.originalScore >= currentThreshold ? 1 : 0;
            }
        });

        updateChartVisuals();
        renderMiniTimeline(); // Met à jour la barre de couleurs

        // Mettre à jour la grille de manière optimisée sans tout détruire
        if (currentSelection.start !== -1) {
            requestAnimationFrame(() => {
                const cards = document.querySelectorAll('.large-card');
                let cardIdx = 0;
                for (let i = currentSelection.start; i <= currentSelection.end; i++) {
                    if (cardIdx < cards.length) {
                        const dp = dataPoints[i];
                        const card = cards[cardIdx];
                        const targetClass = `large-card ${dp.label === 1 ? 'polluted' : ''}`;
                        if (card.className !== targetClass) {
                            card.className = targetClass;
                            const lbl = card.querySelector('.card-label');
                            if (lbl) lbl.innerHTML = dp.label === 1 ? '■ POLLUÉ' : '□ PROPRE';
                        }
                        cardIdx++;
                    }
                }
            });
        }
    }
}

thresholdSlider.addEventListener('input', (e) => applyThresholdChange(e.target.value));
thresholdVal.addEventListener('change', (e) => applyThresholdChange(e.target.value));

// --- 2. NOUVEL IMPORT (AVEC IA) ---
async function handleImportNative(event) {
    if (!window.pywebview || !window.pywebview.api) {
        alert("Mode natif non détecté.");
        return;
    }

    if (!currentWorkspace) {
        alert("Configuration Initiale :\n\nVeuillez d'abord choisir le répertoire de travail où seront enregistrées toutes vos images.");
        return;
    }


    let sourceDir = await window.pywebview.api.open_folder_dialog("Sélectionner le dossier de la carte SD");
    if (!sourceDir) return;

    let riverName = prompt("Nom de la Rivière (ex: Avril, Ziplo, Aire...) :");
    if (!riverName || riverName.trim() === "") return;

    let pov = prompt("Numéro de Point de Vue (ex: 1, 2, 3...) :");
    if (!pov || pov.trim() === "") return;

    // Affichage interface chargement
    emptyState.style.display = 'none';
    mainInterface.style.display = 'flex';
    exportBtn.disabled = true;

    const loadingScreen = document.getElementById('loading-screen');
    const loadingCount = document.getElementById('loading-count');
    if (loadingScreen && loadingCount) {
        loadingScreen.style.display = 'flex';
        loadingCount.textContent = "... Copie & Inférence IA en cours ...";
    }

    let predictions = [];
    try {
        const response = await fetch('http://127.0.0.1:5000/import_and_predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                source_dir: sourceDir,
                workspace_dir: currentWorkspace,
                river: riverName,
                pov: pov
            })
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.error || "Erreur serveur API");
        }
        const data = await response.json();
        predictions = data.predictions;
        window.currentDestDir = data.dest_dir;

        // Afficher le dossier actuel dans la nav
        currentFolderDisplay.value = data.dest_dir;

    } catch (e) {
        console.error("Erreur Backend :", e);
        alert("L'importation ou l'analyse des images a échoué.\nMessage d'erreur: " + e.message);
        if (loadingScreen) loadingScreen.style.display = 'none';
        emptyState.style.display = 'flex';
        mainInterface.style.display = 'none';
        return;
    }

    // Filtrer les nuits et convertir
    let validPredictions = predictions.filter(p => p.status !== "night");
    const total = validPredictions.length;

    dataPoints = validPredictions.map((pred, i) => {
        let cleanPath = pred.path.startsWith('/') ? pred.path.substring(1) : pred.path;
        return {
            id: i,
            name: pred.name,
            date: pred.date,
            time: pred.time,
            shortDate: pred.date !== "N/A" ? pred.date.split('/')[0] : "N/A",
            shortTime: pred.time !== "N/A" ? pred.time.split(':').slice(0, 2).join(':') : "N/A",
            url: `http://127.0.0.1:5000/image/${cleanPath}`,
            path: pred.path,
            originalScore: pred.score,
            label: pred.label,
            status: pred.status
        };
    });

    if (loadingScreen) loadingScreen.style.display = 'none';
    exportBtn.disabled = false;
    miniTimelineContainer.style.display = 'block';

    if (total > 0) {
        const periodEl = document.getElementById('analysis-period');
        if (periodEl) {
            const first = dataPoints[0];
            const last = dataPoints[dataPoints.length - 1];
            periodEl.textContent = `Période : ${first.date} ${first.shortTime} au ${last.date} ${last.shortTime}`;
        }
        initSliders(total);
        renderMiniTimeline();
        initScrubber();
        renderChart();
    } else {
        alert("Aucune image valide trouvée (ex: que des images de nuit).");
    }
}

// --- MINI TIMELINE (Heatmap Scrubber) ---
function renderMiniTimeline() {
    if (miniTrack.children.length !== dataPoints.length) {
        miniTrack.innerHTML = '';
        dataPoints.forEach((dp, i) => {
            const item = document.createElement('div');
            item.className = `mini-item ${dp.label === 1 ? 'polluted' : ''}`;
            item.id = `mini-${i}`;
            miniTrack.appendChild(item);
        });
    } else {
        const children = miniTrack.children;
        for (let i = 0; i < dataPoints.length; i++) {
            const dp = dataPoints[i];
            const targetClass = `mini-item ${dp.label === 1 ? 'polluted' : ''}`;
            if (children[i].className !== targetClass) {
                children[i].className = targetClass;
            }
        }
    }
}

function initScrubber() {
    scrubberWrapper.onmousemove = (e) => {
        if (dataPoints.length === 0) return;

        const rect = scrubberWrapper.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const ratio = Math.max(0, Math.min(1, mouseX / rect.width));
        const index = Math.round(ratio * (dataPoints.length - 1));

        const dp = dataPoints[index];
        if (!dp) return;

        scrubPreview.classList.add('active');

        const previewWidth = 420;
        const previewHeight = 260;
        let leftPos = mouseX;

        if (leftPos < previewWidth / 2) leftPos = previewWidth / 2;
        if (leftPos > rect.width - previewWidth / 2) leftPos = rect.width - previewWidth / 2;
        scrubPreview.style.left = `${leftPos}px`;

        if (rect.top < previewHeight + 40) {
            scrubPreview.style.bottom = 'auto';
            scrubPreview.style.top = '50px';
        } else {
            scrubPreview.style.top = 'auto';
            scrubPreview.style.bottom = '60px';
        }

        const currentUrl = previewImg.getAttribute('data-url');
        if (currentUrl !== dp.url) {
            previewImg.src = dp.url;
            previewImg.setAttribute('data-url', dp.url);
        }
        
        previewInfo.innerHTML = `[${index + 1}/${dataPoints.length}] ${dp.name}`;

        scrubberWrapper.style.setProperty('--cursor-x', `${mouseX}px`);
    };

    scrubberWrapper.onmouseleave = () => {
        scrubPreview.classList.remove('active');
    };

    scrubberWrapper.onclick = (e) => {
        const rect = scrubberWrapper.getBoundingClientRect();
        const ratio = (e.clientX - rect.left) / rect.width;
        const index = Math.round(ratio * (dataPoints.length - 1));
        selectZone(index);
    };
}

// --- DOUBLE SLIDER LOGIC ---
function initSliders(total) {
    sliderMin.max = total - 1;
    sliderMax.max = total - 1;
    sliderMin.value = 0;
    sliderMax.value = total - 1;
    rangeWrapper.style.display = 'flex';

    const updateFromSlider = (e) => {
        let minVal = parseInt(sliderMin.value);
        let maxVal = parseInt(sliderMax.value);

        if (minVal > maxVal) {
            if (e && e.target === sliderMin) {
                sliderMin.value = maxVal;
                minVal = maxVal;
            } else {
                sliderMax.value = minVal;
                maxVal = minVal;
            }
        }

        currentSelection = { start: minVal, end: maxVal };
        updateChartVisuals();

        const count = maxVal - minVal + 1;
        zoneTitle.textContent = `[ ${count} IMG SÉLECTIONNÉES ]`;
        zoneTitle.style.color = '#e4e4e7';
        zoneSection.style.display = 'flex';
        
        clearTimeout(gridTimeout);
        gridTimeout = setTimeout(() => {
            renderGrid(minVal, maxVal);
        }, 50);

        const pctMin = (minVal / (total - 1)) * 100;
        const pctMax = (maxVal / (total - 1)) * 100;
        document.getElementById('slider-track').style.background = `linear-gradient(to right, #cbd5e1 ${pctMin}%, #0f172a ${pctMin}%, #0f172a ${pctMax}%, #cbd5e1 ${pctMax}%)`;
    };

    sliderMin.oninput = updateFromSlider;
    sliderMax.oninput = updateFromSlider;
    updateFromSlider(null);
}

// --- PLUGINS CHART.JS ---
const sliderAlignmentPlugin = {
    id: 'sliderAlignment',
    afterLayout: (chart) => {
        const rangeWrap = document.getElementById('range-wrapper');
        const miniWrap = document.getElementById('scrubber-wrapper');
        const containerWrap = document.getElementById('mini-timeline-container');
        if (!rangeWrap || !miniWrap) return;
        const { left, width } = chart.chartArea;
        rangeWrap.style.marginLeft = `${left}px`;
        rangeWrap.style.width = `${width}px`;
        miniWrap.style.marginLeft = `${left}px`;
        miniWrap.style.width = `${width}px`;
        if (containerWrap) containerWrap.style.width = '100%';
    }
};

const highlightPlugin = {
    id: 'highlightPollution',
    beforeDraw: (chart) => {
        if (!dataPoints || dataPoints.length === 0) return;
        const { ctx, chartArea: { top, bottom, left, right }, scales: { x } } = chart;
        ctx.save();

        // Zones polluées
        ctx.fillStyle = COLOR_POLLUTED_BG;
        let startX = null;
        for (let i = 0; i < dataPoints.length; i++) {
            if (dataPoints[i].label === 1) {
                if (startX === null) {
                    startX = Math.max(left, x.getPixelForValue(i) - (x.width / dataPoints.length) / 2);
                }
                if (i === dataPoints.length - 1) {
                    const endX = Math.min(right, x.getPixelForValue(i) + (x.width / dataPoints.length) / 2);
                    ctx.fillRect(startX, top, endX - startX, bottom - top);
                }
            } else {
                if (startX !== null) {
                    const endX = Math.min(right, x.getPixelForValue(i) - (x.width / dataPoints.length) / 2);
                    ctx.fillRect(startX, top, endX - startX, bottom - top);
                    startX = null;
                }
            }
        }

        // Zone sélectionnée manuellement
        if (currentSelection.start !== -1 && currentSelection.end !== -1) {
            ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
            const selStartX = Math.max(left, x.getPixelForValue(currentSelection.start) - (x.width / dataPoints.length) / 2);
            const selEndX = Math.min(right, x.getPixelForValue(currentSelection.end) + (x.width / dataPoints.length) / 2);
            ctx.fillRect(selStartX, top, selEndX - selStartX, bottom - top);
        }
        ctx.restore();
    }
};

const daySeparatorPlugin = {
    id: 'daySeparator',
    beforeDraw: (chart) => {
        if (!dataPoints || dataPoints.length === 0) return;
        const { ctx, chartArea: { top, bottom }, scales: { x } } = chart;
        ctx.save();
        ctx.strokeStyle = '#64748b';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);

        let lastDate = null;
        for (let i = 0; i < dataPoints.length; i++) {
            const dp = dataPoints[i];
            if (lastDate !== null && dp.date !== lastDate) {
                const xPos = x.getPixelForValue(i) - (x.width / dataPoints.length) / 2;
                ctx.beginPath();
                ctx.moveTo(xPos, top);
                ctx.lineTo(xPos, bottom);
                ctx.stroke();

                ctx.fillStyle = '#64748b';
                ctx.font = '10px "JetBrains Mono", monospace';
                ctx.fillText(dp.date, xPos + 6, top + 15);
            }
            lastDate = dp.date;
        }
        ctx.restore();
    }
};

const axisSyncPlugin = {
    id: 'axisSyncPlugin',
    afterDraw: (chart) => {
        if (!dataPoints || dataPoints.length === 0) return;
        const axisWrapper = document.getElementById('timeline-axis');
        if (!axisWrapper) return;
        axisWrapper.innerHTML = '';
        const { scales: { x }, chartArea: { left: areaLeft } } = chart;

        let lastDate = null;
        for (let i = 0; i < dataPoints.length; i++) {
            const dp = dataPoints[i];
            if (lastDate !== null && lastDate !== dp.date) {
                const xPos = x.getPixelForValue(i) - (x.width / dataPoints.length) / 2 - areaLeft;

                const tick = document.createElement('div');
                tick.className = 'axis-tick';
                tick.style.left = `${xPos}px`;
                tick.innerText = dp.date;

                const line = document.createElement('div');
                line.className = 'axis-line';
                line.style.left = `${xPos}px`;

                axisWrapper.appendChild(tick);
                axisWrapper.appendChild(line);
            }
            lastDate = dp.date;
        }
    }
};

Chart.register(highlightPlugin, sliderAlignmentPlugin, daySeparatorPlugin, axisSyncPlugin);

// --- RENDER CHART ---
function renderChart() {
    const ctx = document.getElementById('pollutionChart').getContext('2d');
    if (chartInstance) chartInstance.destroy();

    chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dataPoints.map(d => d.time === 'N/A' ? d.name : d.shortTime),
            datasets: [{
                label: "CONFIANCE IA (%)",
                data: dataPoints.map(d => d.manualOverride ? d.label * 100 : d.originalScore * 100),
                borderColor: '#0f172a',
                borderWidth: 2,
                pointRadius: 0,
                pointHoverRadius: 6,
                pointHoverBackgroundColor: '#0f172a',
                tension: 0.2,
                fill: false,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            layout: { padding: { left: 16, right: 16, top: 10, bottom: 0 } },
            interaction: { mode: 'nearest', axis: 'x', intersect: false },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(255, 255, 255, 0.95)',
                    titleColor: '#0f172a',
                    bodyColor: '#0f172a',
                    titleFont: { family: "'JetBrains Mono', monospace", size: 11 },
                    bodyFont: { family: "'JetBrains Mono', monospace", size: 11 },
                    borderColor: '#cbd5e1',
                    borderWidth: 1,
                    callbacks: {
                        label: function (context) {
                            const dp = dataPoints[context.dataIndex];
                            return [
                                `SCORE: ${(dp.originalScore * 100).toFixed(1)}%`,
                                `STATE: ${dp.label === 1 ? "POLLUÉ [!]" : "PROPRE [OK]"}`
                            ];
                        }
                    }
                }
            },
            scales: {
                x: {
                    display: true,
                    ticks: { color: '#475569', font: { family: "'JetBrains Mono', monospace", size: 10 }, maxTicksLimit: 20 },
                    grid: { color: '#cbd5e1' }
                },
                y: {
                    min: 0, max: 105,
                    ticks: { color: '#475569', font: { family: "'JetBrains Mono', monospace", size: 10 } },
                    grid: { color: '#cbd5e1' }
                }
            },
            onClick: (event, elements, chart) => {
                let index = -1;
                if (elements.length > 0) {
                    index = elements[0].index;
                } else {
                    const canvasPosition = Chart.helpers.getRelativePosition(event, chart);
                    const xValue = chart.scales.x.getValueForPixel(canvasPosition.x);
                    if (xValue !== undefined) index = Math.round(xValue);
                }
                if (index >= 0 && index < dataPoints.length) selectZone(index);
            }
        }
    });
}

function updateChartVisuals() {
    if (chartInstance) {
        chartInstance.data.datasets[0].data = dataPoints.map(d => d.manualOverride ? d.label * 100 : d.originalScore * 100);
        chartInstance.update();
    }
}

// --- SÉLECTION & GRILLE ---
function selectZone(index) {
    const targetLabel = dataPoints[index].label;

    let start = index;
    while (start > 0 && dataPoints[start - 1].label === targetLabel) start--;

    let end = index;
    while (end < dataPoints.length - 1 && dataPoints[end + 1].label === targetLabel) end++;

    currentSelection = { start, end };
    updateChartVisuals();

    const count = end - start + 1;
    zoneTitle.textContent = `[ ${count} IMG - ${targetLabel === 1 ? 'ALERTE' : 'NOMINAL'} ]`;
    zoneTitle.style.color = targetLabel === 1 ? COLOR_POLLUTED : COLOR_CLEAN;

    zoneSection.style.display = 'flex';

    if (rangeWrapper.style.display === 'flex' || rangeWrapper.style.display === '') {
        sliderMin.value = start;
        sliderMax.value = end;
        const pctMin = (start / (dataPoints.length - 1)) * 100;
        const pctMax = (end / (dataPoints.length - 1)) * 100;
        document.getElementById('slider-track').style.background = `linear-gradient(to right, #cbd5e1 ${pctMin}%, #0f172a ${pctMin}%, #0f172a ${pctMax}%, #cbd5e1 ${pctMax}%)`;
    }

    clearTimeout(gridTimeout);
    gridTimeout = setTimeout(() => {
        renderGrid(start, end);
        zoneSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 50);
}

function renderGrid(start, end) {
    imageGrid.innerHTML = '';
    for (let i = start; i <= end; i++) {
        const dp = dataPoints[i];

        const card = document.createElement('div');
        card.className = `large-card ${dp.label === 1 ? 'polluted' : ''}`;

        const lbl = document.createElement('div');
        lbl.className = 'card-label';
        lbl.innerHTML = dp.label === 1 ? '■ POLLUÉ' : '□ PROPRE';

        const img = document.createElement('img');
        img.src = dp.url;
        img.alt = dp.name;
        img.loading = "lazy";

        const expBtn = document.createElement('button');
        expBtn.className = 'expand-btn';
        expBtn.innerHTML = '[ ZOOM ]';
        expBtn.onclick = (e) => {
            e.stopPropagation();
            showModal(dp.url);
        };

        card.appendChild(img);
        card.appendChild(lbl);
        card.appendChild(expBtn);

        card.addEventListener('click', () => { toggleLabel(i); });
        imageGrid.appendChild(card);
    }
}

function toggleLabel(index) {
    const dp = dataPoints[index];
    dp.label = dp.label === 1 ? 0 : 1;

    // NOUVEAU : On marque l'image comme "forcée" par l'utilisateur
    dp.manualOverride = true;

    const miniItem = document.getElementById(`mini-${index}`);
    if (miniItem) {
        if (dp.label === 1) miniItem.classList.add('polluted');
        else miniItem.classList.remove('polluted');
    }

    updateChartVisuals();
    renderGrid(currentSelection.start, currentSelection.end);
}

// --- LOGIQUE BULK ---
const bulkCleanBtn = document.getElementById('bulk-clean-btn');
const bulkPollutedBtn = document.getElementById('bulk-polluted-btn');

function applyBulkLabel(labelVal) {
    if (currentSelection.start === -1 || currentSelection.end === -1) return;
    for (let i = currentSelection.start; i <= currentSelection.end; i++) {
        const dp = dataPoints[i];
        dp.label = labelVal;
        dp.manualOverride = true;
        const miniItem = document.getElementById(`mini-${i}`);
        if (miniItem) {
            if (labelVal === 1) miniItem.classList.add('polluted');
            else miniItem.classList.remove('polluted');
        }
    }
    updateChartVisuals();
    renderGrid(currentSelection.start, currentSelection.end);
}

if (bulkCleanBtn && bulkPollutedBtn) {
    bulkCleanBtn.addEventListener('click', () => applyBulkLabel(0));
    bulkPollutedBtn.addEventListener('click', () => applyBulkLabel(1));
}
// --- SAUVEGARDE / EXPORT EXIF ---
async function handleExport() {
    if (dataPoints.length === 0 || !window.currentDestDir) return;

    const exportBtnInstance = document.getElementById('export-btn');
    const originalText = exportBtnInstance.innerHTML;
    exportBtnInstance.innerHTML = "[!] ÉCRITURE EXIF...";
    exportBtnInstance.disabled = true;

    try {
        const payload = {
            dest_dir: window.currentDestDir,
            labels: dataPoints.map(dp => ({
                name: dp.name,
                path: dp.path,
                date: dp.date,
                time: dp.time,
                score: dp.originalScore,
                ai_label: dp.originalScore >= currentThreshold ? 1 : 0,
                label: dp.label,
                status: dp.status
            }))
        };

        const response = await fetch('http://127.0.0.1:5000/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error("Erreur système de sauvegarde");

        exportBtnInstance.innerHTML = "[OK] DONNÉES SAUVEGARDÉES";
        setTimeout(() => {
            exportBtnInstance.innerHTML = originalText;
            exportBtnInstance.disabled = false;
        }, 4000);

    } catch (e) {
        console.error(e);
        alert("Erreur lors de la sauvegarde: " + e.message);
        exportBtnInstance.innerHTML = originalText;
        exportBtnInstance.disabled = false;
    }
}