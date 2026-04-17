// DOM Elements
const importInput = document.getElementById('import-input');
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

closeModal.onclick = () => { modal.style.display = 'none'; };
modal.onclick = (e) => { if(e.target === modal) modal.style.display = 'none'; };

function showModal(url) {
    modalImg.src = url;
    modal.style.display = 'flex';
}

// State
let dataPoints = [];
let chartInstance = null;
let currentSelection = { start: -1, end: -1 };

// Palette (doit correspondre au CSS)
const COLOR_CLEAN = '#06b6d4';     // Cyan
const COLOR_POLLUTED = '#f97316';  // Orange

// --- INITIALISATION ---
importInput.addEventListener('change', handleImport);
exportBtn.addEventListener('click', handleExport);

// --- UTILITAIRE DATE/HEURE ---
function extractDateTime(filename) {
    const match = filename.match(/^(\d{2})(\d{2})(\d{4})_(\d{2})(\d{2})(\d{2})/);
    if (match) {
        return {
            date: `${match[1]}/${match[2]}/${match[3]}`,
            time: `${match[4]}:${match[5]}:${match[6]}`
        };
    }
    return { date: 'N/A', time: 'N/A' };
}

// --- IMPORTATION ---
async function handleImport(event) {
    let rawFiles = Array.from(event.target.files).filter(file => file.type.startsWith('image/'));
    if (rawFiles.length === 0) return;

    // --- INTERCEPTION : Renommage automatique SD ---
    let files = [];
    const needsRename = rawFiles.some(file => !/^\d{8}_\d{6}_/.test(file.name));
    
    if (needsRename) {
        const riverName = prompt("De quelle rivière proviennent ces images ? (ex: Avril, Ziplo, Aire, etc.)");
        if (!riverName || riverName.trim() === "") {
            event.target.value = ''; // Reset input
            return;
        }

        const cleanRiverName = riverName.trim().charAt(0).toUpperCase() + riverName.trim().slice(1).toLowerCase();

        files = rawFiles.map(file => {
            if (/^\d{8}_\d{6}_/.test(file.name)) {
                return file; // Déjà formaté correctement
            }

            // Extraction Date & Heure depuis la carte SD
            const d = new Date(file.lastModified || Date.now());
            const day = String(d.getDate()).padStart(2, '0');
            const month = String(d.getMonth() + 1).padStart(2, '0');
            const year = d.getFullYear();
            const hours = String(d.getHours()).padStart(2, '0');
            const mins = String(d.getMinutes()).padStart(2, '0');
            const secs = String(d.getSeconds()).padStart(2, '0');

            const lastDot = file.name.lastIndexOf('.');
            const originalNameSansExt = lastDot !== -1 ? file.name.substring(0, lastDot) : file.name;
            const extension = lastDot !== -1 ? file.name.substring(lastDot) : '.jpg';

            // Formatage: DDMMYYYY_HHMMSS_ORIGINALNAME_RiverName.jpg
            const newName = `${day}${month}${year}_${hours}${mins}${secs}_${originalNameSansExt}_${cleanRiverName}${extension}`;
            return new File([file], newName, { type: file.type, lastModified: file.lastModified });
        });
    } else {
        files = rawFiles;
    }

    files.sort((a, b) => a.name.localeCompare(b.name));
    
    // Afficher l'écran de chargement
    emptyState.style.display = 'none';
    mainInterface.style.display = 'flex';
    exportBtn.disabled = true;
    
    const loadingScreen = document.getElementById('loading-screen');
    const loadingCount = document.getElementById('loading-count');
    if(loadingScreen && loadingCount) {
        loadingScreen.style.display = 'flex';
        loadingCount.textContent = files.length;
    }

    // Préparer les images pour le Backend Python (API)
    const formData = new FormData();
    files.forEach(file => formData.append('images', file));

    let predictions = [];
    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) throw new Error("Erreur serveur API");
        const data = await response.json();
        predictions = data.predictions;
    } catch (e) {
        console.error("Erreur Backend :", e);
        alert("⚠️ Modèle IA injoignable (Avez-vous bien lancé `python UI/server.py` dans le terminal ?).\n\nGénération de valeurs aléatoires pour la démonstration.");
        // Fallback aléatoire en cas d'erreur
        predictions = files.map((f, i) => {
            const isPolluted = i > 15 && i < 30;
            return {
                name: f.name,
                score: isPolluted ? 0.7 + Math.random() * 0.2 : Math.random() * 0.2,
                label: isPolluted ? 1 : 0
            };
        });
    }

    // Filtrer les images de nuit avant de les mettre dans le DOM
    let validFiles = [];
    let validPredictions = [];
    
    files.forEach(file => {
        const pred = predictions.find(p => p.name === file.name);
        if (pred && pred.status !== "night") {
            validFiles.push(file);
            validPredictions.push(pred);
        }
    });

    const total = validFiles.length;
    dataPoints = validFiles.map((file, i) => {
        const pred = validPredictions.find(p => p.name === file.name);
        const dt = extractDateTime(file.name);
        return {
            id: i,
            name: file.name,
            date: dt.date,
            time: dt.time,
            url: URL.createObjectURL(file), 
            originalScore: pred ? pred.score : 0.0,
            label: pred ? pred.label : 0
        };
    });

    // Masquer le chargement et afficher le tableau
    if(loadingScreen) loadingScreen.style.display = 'none';
    exportBtn.disabled = false;
    miniTimelineContainer.style.display = 'block';

    initSliders(total);
    renderMiniTimeline();
    initScrubber();
    renderChart();
}

// --- MINI TIMELINE (Heatmap Scrubber) ---
function renderMiniTimeline() {
    miniTrack.innerHTML = '';
    dataPoints.forEach((dp, i) => {
        const item = document.createElement('div');
        item.className = `mini-item ${dp.label === 1 ? 'polluted' : ''}`;
        item.id = `mini-${i}`;
        miniTrack.appendChild(item);
    });
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

        // Afficher l'aperçu
        scrubPreview.classList.add('active');
        
        // --- GESTION DES BORDS POUR L'APERÇU ---
        const previewWidth = 500;
        const previewHeight = 320;
        let leftPos = mouseX;
        
        // Empêcher de sortir à gauche/droite
        if (leftPos < previewWidth / 2) leftPos = previewWidth / 2;
        if (leftPos > rect.width - previewWidth / 2) leftPos = rect.width - previewWidth / 2;
        scrubPreview.style.left = `${leftPos}px`;

        // Éviter le rognage en haut : si on est trop haut dans la page, on affiche en dessous
        if (rect.top < previewHeight + 40) {
            scrubPreview.style.bottom = 'auto';
            scrubPreview.style.top = '70px'; // Un peu plus bas pour ne pas toucher la barre
        } else {
            scrubPreview.style.top = 'auto';
            scrubPreview.style.bottom = '80px';
        }

        previewImg.src = dp.url;
        previewInfo.innerHTML = `${index + 1} - ${dp.name}`;

        // Indicateur vertical (déplacement via ::before style)
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
            if (e.target === sliderMin) {
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
        zoneTitle.textContent = `${count} image(s) sélectionnée(s) manuellement`;
        zoneTitle.style.color = '#f8fafc';
        zoneSection.style.display = 'flex';
        renderGrid(minVal, maxVal);
        
        // Mettre en couleur la zone sélectionnée sur la piste (track)
        const pctMin = (minVal / (total - 1)) * 100;
        const pctMax = (maxVal / (total - 1)) * 100;
        document.getElementById('slider-track').style.background = `linear-gradient(to right, rgba(255,255,255,0.1) ${pctMin}%, var(--color-clean) ${pctMin}%, var(--color-clean) ${pctMax}%, rgba(255,255,255,0.1) ${pctMax}%)`;
    };

    sliderMin.oninput = updateFromSlider;
    sliderMax.oninput = updateFromSlider;
    
    // Initialisation de la couleur
    updateFromSlider({target: null});
}

// --- PLUGIN CHART.JS POUR ALIGNER LE SLIDER ET LA TIMELINE ---
const sliderAlignmentPlugin = {
    id: 'sliderAlignment',
    afterLayout: (chart) => {
        const rangeWrap = document.getElementById('range-wrapper');
        const miniWrap = document.getElementById('scrubber-wrapper');
        const containerWrap = document.getElementById('mini-timeline-container');
        
        if (!rangeWrap || !miniWrap) return;
        
        const { left, width } = chart.chartArea;
        
        // Aligner précisément sur la gauche de la zone de dessin Chart.js
        rangeWrap.style.marginLeft = `${left}px`;
        rangeWrap.style.width = `${width}px`;
        
        miniWrap.style.marginLeft = `${left}px`;
        miniWrap.style.width = `${width}px`;
        
        if (containerWrap) containerWrap.style.width = '100%';
    }
};

// --- PLUGIN CHART.JS POUR LE FOND (HIGHLIGHT) ---
const highlightPlugin = {
    id: 'highlightPollution',
    beforeDraw: (chart) => {
        if (!dataPoints || dataPoints.length === 0) return;
        const { ctx, chartArea: { top, bottom, left, right }, scales: { x } } = chart;
        
        ctx.save();
        
        // 1) Dessiner les zones de pollution (label == 1)
        ctx.fillStyle = 'rgba(249, 115, 22, 0.2)'; // Orange
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
        
        // 2) Dessiner la zone actuellement SELECTIONNEE 
        if (currentSelection.start !== -1 && currentSelection.end !== -1) {
            ctx.fillStyle = 'rgba(255, 255, 255, 0.15)'; // Surbrillance blanche claire
            const selStartX = Math.max(left, x.getPixelForValue(currentSelection.start) - (x.width / dataPoints.length) / 2);
            const selEndX = Math.min(right, x.getPixelForValue(currentSelection.end) + (x.width / dataPoints.length) / 2);
            ctx.fillRect(selStartX, top, selEndX - selStartX, bottom - top);
        }
        
        ctx.restore();
    }
};

// --- PLUGIN CHART.JS POUR DELIMITER LES JOURS ---
const daySeparatorPlugin = {
    id: 'daySeparator',
    beforeDraw: (chart) => {
        if (!dataPoints || dataPoints.length === 0) return;
        const { ctx, chartArea: { top, bottom }, scales: { x } } = chart;
        
        ctx.save();
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);

        let lastDate = null;
        for (let i = 0; i < dataPoints.length; i++) {
            const dp = dataPoints[i];
            // Changement de date détecté
            if (lastDate !== null && dp.date !== lastDate) {
                // On met le trait exactement au milieu entre la dernière image du jour et la première du jour suivant
                const xPos = x.getPixelForValue(i) - (x.width / dataPoints.length) / 2;
                
                // Ligne pointillée
                ctx.beginPath();
                ctx.moveTo(xPos, top);
                ctx.lineTo(xPos, bottom);
                ctx.stroke();

                // Texte de la nouvelle date
                ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
                ctx.font = 'bold 11px Arial';
                ctx.fillText(dp.date, xPos + 6, top + 15);
            }
            lastDate = dp.date;
        }
        ctx.restore();
    }
};

// --- PLUGIN CHART.JS POUR L'AXE X DE LA TIMELINE ---
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
                // Position relative vu que #scrubber-wrapper a marginLeft = areaLeft
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

// --- GRAPHIQUE (CHART.JS) ---
function renderChart() {
    const ctx = document.getElementById('pollutionChart').getContext('2d');
    
    if (chartInstance) {
        chartInstance.destroy();
    }

    chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dataPoints.map(d => d.time === 'N/A' ? d.name : d.time),
            datasets: [{
                label: "Confiance du Modèle (%)",
                data: dataPoints.map(d => d.originalScore * 100), // En pourcentage
                borderColor: COLOR_CLEAN, // La ligne est constante (score modèle)
                borderWidth: 2,
                pointRadius: 0, // Pour une courbe propre
                pointHoverRadius: 6,
                tension: 0.3, // "Smoothness"
                fill: false,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            layout: {
                padding: {
                    left: 0,
                    right: 0, // Suppression de la marge à droite
                    top: 10,
                    bottom: 0
                }
            },
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false,
            },
            plugins: {
                legend: {
                    display: true,
                    labels: { color: '#f8fafc' }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const dp = dataPoints[context.dataIndex];
                            return [
                                `Confiance: ${(dp.originalScore*100).toFixed(1)}%`,
                                `Statut Actuel: ${dp.label === 1 ? "POLLUÉ 🟠" : "PROPRE 💧"}`
                            ];
                        }
                    }
                }
            },
            scales: {
                x: {
                    display: true,
                    title: { display: true, text: 'Heure', color: '#94a3b8' },
                    ticks: { color: '#94a3b8', maxTicksLimit: 20 },
                    grid: { color: 'rgba(255,255,255,0.05)' }
                },
                y: {
                    min: 0,
                    max: 105,
                    title: { display: true, text: 'Confiance Modèle (%)', color: '#94a3b8' },
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(255,255,255,0.05)' }
                }
            },
            onClick: (event, elements, chart) => {
                let index = -1;
                if (elements.length > 0) {
                    index = elements[0].index;
                } else {
                    // Si clic en dehors d'un point exact
                    const canvasPosition = Chart.helpers.getRelativePosition(event, chart);
                    const xValue = chart.scales.x.getValueForPixel(canvasPosition.x);
                    if (xValue !== undefined) index = Math.round(xValue);
                }
                
                if (index >= 0 && index < dataPoints.length) {
                    selectZone(index);
                }
            }
        }
    });
}

function updateChartVisuals() {
    if (chartInstance) {
        chartInstance.update();
    }
}

// --- LOGIQUE DE SÉLECTION DE ZONE ---
function selectZone(index) {
    const targetLabel = dataPoints[index].label;
    
    // Trouver le bloc continu ayant le même label
    let start = index;
    while (start > 0 && dataPoints[start - 1].label === targetLabel) {
        start--;
    }
    
    let end = index;
    while (end < dataPoints.length - 1 && dataPoints[end + 1].label === targetLabel) {
        end++;
    }
    
    currentSelection = { start, end };
    updateChartVisuals(); 
    
    const count = end - start + 1;
    zoneTitle.textContent = `${count} image(s) ${targetLabel === 1 ? '(Pollution)' : '(Propres)'}`;
    zoneTitle.style.color = targetLabel === 1 ? COLOR_POLLUTED : COLOR_CLEAN;

    zoneSection.style.display = 'flex';
    
    // Sync slider thumbs
    if (rangeWrapper.style.display === 'flex' || rangeWrapper.style.display === '') {
        sliderMin.value = start;
        sliderMax.value = end;
        
        const pctMin = (start / (dataPoints.length - 1)) * 100;
        const pctMax = (end / (dataPoints.length - 1)) * 100;
        document.getElementById('slider-track').style.background = `linear-gradient(to right, rgba(255,255,255,0.1) ${pctMin}%, var(--color-clean) ${pctMin}%, var(--color-clean) ${pctMax}%, rgba(255,255,255,0.1) ${pctMax}%)`;
    }

    renderGrid(start, end);
    // Petit scroll doux vers la section de la grille
    zoneSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// --- GRILLE D'IMAGES (LARGE) ---
function renderGrid(start, end) {
    imageGrid.innerHTML = '';
    
    for (let i = start; i <= end; i++) {
        const dp = dataPoints[i];
        
        const card = document.createElement('div');
        card.className = `large-card ${dp.label === 1 ? 'polluted' : ''}`;
        
        // Label superposé
        const lbl = document.createElement('div');
        lbl.className = 'card-label';
        lbl.innerHTML = dp.label === 1 ? '🟠 POLLUÉ' : '💧 PROPRE';
        
        const img = document.createElement('img');
        img.src = dp.url;
        img.alt = dp.name;
        img.loading = "lazy";

        // Bouton pour agrandir
        const expBtn = document.createElement('button');
        expBtn.className = 'expand-btn';
        expBtn.innerHTML = '🔍';
        expBtn.onclick = (e) => {
            e.stopPropagation(); // Evite de déclencher toggleLabel
            showModal(dp.url);
        };

        card.appendChild(img);
        card.appendChild(lbl);
        card.appendChild(expBtn);
        
        card.addEventListener('click', () => {
            toggleLabel(i);
        });

        imageGrid.appendChild(card);
    }
}

function toggleLabel(index) {
    const dp = dataPoints[index];
    dp.label = dp.label === 1 ? 0 : 1; 
    
    // Mettre à jour l'indicateur dans la mini-timeline (Scrubber)
    const miniItem = document.getElementById(`mini-${index}`);
    if (miniItem) {
        if (dp.label === 1) miniItem.classList.add('polluted');
        else miniItem.classList.remove('polluted');
    }

    // Mise à jour de la charte graphique en direct (changement du fond)
    updateChartVisuals();
    
    // Rafraîchir la grille (on conserve les bornes de la sélection, même si la zone continue est brisée)
    renderGrid(currentSelection.start, currentSelection.end);
}

// --- EXPORTATION ---
function handleExport() {
    if (dataPoints.length === 0) return;

    let csvContent = "Date,Heure,Nom_Image,Confidence_Modele,Label_Utilisateur\n";
    
    dataPoints.forEach(row => {
        csvContent += `${row.date},${row.time},${row.name},${row.originalScore.toFixed(4)},${row.label}\n`;
    });

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", "waterwatcher_labels.csv");
    document.body.appendChild(link);
    
    link.click();
    document.body.removeChild(link);
}
