console.log("WaterWatcher V1 : Script app.js chargé !");
// Les éléments seront récupérés dans le DOMContentLoaded pour plus de sécurité.
let importBtn, exportBtn, emptyState, mainInterface, zoneSection, imageGrid, zoneTitle;
let miniTimelineContainer, miniTrack, scrubberWrapper, scrubPreview, previewImg, previewInfo;
let sliderMin, sliderMax, rangeWrapper, modal, modalImg, closeModal;

function showModal(url) {
    modalImg.src = url;
    modal.style.display = 'flex';
}

// State
let dataPoints = [];
let chartInstance = null;
let currentSelection = { start: -1, end: -1 };

// Palette (doit correspondre au CSS)
// const COLOR_CLEAN = '#06b6d4';     // Cyan
// const COLOR_POLLUTED = '#f97316';  // Orange

const COLOR_CLEAN = '#00f0ff';     // Cyan technique
const COLOR_POLLUTED = '#ff2a55';  // Rouge d'alerte technique

// --- INITIALISATION ---
document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM prêt dans WaterWatcher V1");
    console.log("DOM Chargé, initialisation des boutons...");
    
    // Récupération des éléments
    importBtn = document.getElementById('import-btn');
    exportBtn = document.getElementById('export-btn');
    console.log("Bouton Import trouvé ?", !!importBtn);
    console.log("Bouton Export trouvé ?", !!exportBtn);
    
    emptyState = document.getElementById('empty-state');
    mainInterface = document.getElementById('main-interface');
    zoneSection = document.getElementById('zone-section');
    imageGrid = document.getElementById('image-grid');
    zoneTitle = document.getElementById('zone-title');
    miniTimelineContainer = document.getElementById('mini-timeline-container');
    miniTrack = document.getElementById('mini-track');
    scrubberWrapper = document.getElementById('scrubber-wrapper');
    scrubPreview = document.getElementById('scrub-preview');
    previewImg = document.getElementById('preview-img');
    previewInfo = document.getElementById('preview-info');
    sliderMin = document.getElementById('slider-min');
    sliderMax = document.getElementById('slider-max');
    rangeWrapper = document.getElementById('range-wrapper');
    modal = document.getElementById('image-modal');
    modalImg = document.getElementById('modal-img');
    closeModal = document.getElementById('close-modal');

    // Gestion Modal
    if (closeModal) closeModal.onclick = () => { modal.style.display = 'none'; };
    if (modal) modal.onclick = (e) => { if(e.target === modal) modal.style.display = 'none'; };

    if (importBtn) {
        importBtn.addEventListener('click', handleImportNative);
        console.log("Bouton Import prêt.");
    } else {
        console.error("Erreur : Bouton import-btn non trouvé !");
    }
    
    if (exportBtn) {
        exportBtn.addEventListener('click', handleExport);
        console.log("Bouton Export prêt.");
    }
});

// --- IMPORTATION NATIVE ---
async function handleImportNative(event) {
    try {
        console.log("Clic sur Import détecté.");
        if (!window.pywebview || !window.pywebview.api) {
            console.error("pywebview.api non disponible.");
            if (importBtn) importBtn.innerHTML = "❌ ERREUR: PyWebView non détecté";
            return;
        }
        
        // V1: Demander uniquement le dossier source (évaluation sur place)
        console.log("Sélection du dossier source.");
        let sourceDir = await window.pywebview.api.open_folder_dialog("Sélectionnez le dossier contenant les images à analyser");
        if (!sourceDir) return;
        
        // 3. Valeurs par défaut pour contourner le bug du prompt natif
        let riverName = "Ziplo";
        let pov = "1";
    
    // Affichage interface chargement
    emptyState.style.display = 'none';
    mainInterface.style.display = 'flex';
    exportBtn.disabled = true;
    
    const loadingScreen = document.getElementById('loading-screen');
    const loadingCount = document.getElementById('loading-count');
    if(loadingScreen && loadingCount) {
        loadingScreen.style.display = 'flex';
        loadingCount.textContent = "... Copie & IA en cours ...";
    }

    let predictions = [];
    try {
        const response = await fetch('/import_and_predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                source_dir: sourceDir,
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
        window.currentDestDir = data.dest_dir; // Utilisé lors du Save EXIF
    } catch (e) {
        console.error("Erreur Backend :", e);
        alert("L'importation ou l'analyse des images a échoué.\nMessage d'erreur: " + e.message);
        if(loadingScreen) loadingScreen.style.display = 'none';
        emptyState.style.display = 'flex';
        mainInterface.style.display = 'none';
        return;
    }

    // Filtrer les nuits et convertir pour le graphique
    let validPredictions = predictions.filter(p => p.status !== "night");
    const total = validPredictions.length;
    
    dataPoints = validPredictions.map((pred, i) => {
        let cleanPath = pred.path.startsWith('/') ? pred.path.substring(1) : pred.path;
        return {
            id: i,
            name: pred.name,
            date: pred.date,
            time: pred.time,
            url: `/image/${cleanPath}`,
            path: pred.path,
            originalScore: pred.score,
            label: pred.label,
            status: pred.status
        };
    });

    if(loadingScreen) loadingScreen.style.display = 'none';
    exportBtn.disabled = false;
    miniTimelineContainer.style.display = 'block';

    if (total > 0) {
        initSliders(total);
        renderMiniTimeline();
        initScrubber();
        renderChart();
    } else {
        console.log("Aucune image valide trouvée.");
    }
    } catch(err) {
        console.error("Erreur critique handleImport:", err);
        if (importBtn) importBtn.innerHTML = "❌ Erreur interne (voir console)";
    }
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

// --- ENREGISTREMENT PLUGINS CHART.JS ---
try {
    if (typeof Chart !== 'undefined') {
        Chart.register(highlightPlugin, sliderAlignmentPlugin, daySeparatorPlugin, axisSyncPlugin);
    }
} catch (e) {
    console.error("Erreur d'enregistrement Chart.js plugins:", e);
}

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

// --- SAUVEGARDE NATIVE ---
async function handleExport() {
    if (dataPoints.length === 0 || !window.currentDestDir) return;

    // Loading UX
    const exportBtnInstance = document.getElementById('export-btn');
    const originalText = exportBtnInstance.innerHTML;
    exportBtnInstance.innerHTML = "Écriture EXIF en cours...";
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
                label: dp.label,
                status: dp.status
            }))
        };
        
        const response = await fetch('/save', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) throw new Error("Erreur de sauvegarde");
        
        exportBtnInstance.innerHTML = "✔️ Labels & CSV Enregistrés!";
        setTimeout(() => {
            exportBtnInstance.innerHTML = originalText;
            exportBtnInstance.disabled = false;
        }, 4000);
        
    } catch(e) {
        console.error(e);
        alert("Erreur lors de la sauvegarde: " + e.message);
        exportBtnInstance.innerHTML = originalText;
        exportBtnInstance.disabled = false;
    }
}
