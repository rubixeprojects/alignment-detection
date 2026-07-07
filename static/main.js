document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const sampleList = document.getElementById("sample-list");
    const originalPlaceholder = document.getElementById("original-placeholder");
    const originalPreview = document.getElementById("original-preview");
    const annotatedPlaceholder = document.getElementById("annotated-placeholder");
    const annotatedPreview = document.getElementById("annotated-preview");
    const dropZone = document.getElementById("drop-zone");
    const fileUpload = document.getElementById("file-upload");
    
    const globalStatus = document.getElementById("global-status");
    const verdictDisplay = document.getElementById("verdict-display");
    const angleDisplay = document.getElementById("angle-display");
    const gaugeNeedle = document.getElementById("gauge-needle");
    
    const centerConfVal = document.getElementById("center-conf-val");
    const centerConfBar = document.getElementById("center-conf-bar");
    const notchConfVal = document.getElementById("notch-conf-val");
    const notchConfBar = document.getElementById("notch-conf-bar");
    
    const detectionsLogBody = document.getElementById("detections-log-body");
    
    const controlsOriginal = document.getElementById("controls-original");
    const controlsAnnotated = document.getElementById("controls-annotated");

    // Camera Integration Elements & State
    const cameraUrlInput = document.getElementById("camera-url");
    const connectBtn = document.getElementById("connect-btn");
    const disconnectBtn = document.getElementById("disconnect-btn");
    let cameraPollingInterval = null;

    // Zoom & Pan State
    let zoomLevel = 1.0;
    let panOffset = { x: 0, y: 0 };
    let isDragging = false;
    let dragStart = { x: 0, y: 0 };

    // Initialize Page
    fetchSamples();
    setupUploadHandlers();
    setupZoomAndPanHandlers();
    setupCameraHandlers();

    // Fetch and populate test samples
    async function fetchSamples() {
        try {
            const res = await fetch("/api/samples");
            const files = await res.json();
            
            sampleList.innerHTML = "";
            if (files.length === 0) {
                sampleList.innerHTML = `<div class="loading-spinner">No samples found.</div>`;
                return;
            }
            
            files.forEach(file => {
                const item = document.createElement("div");
                item.className = "sample-item";
                item.innerHTML = `
                    <span class="sample-icon">🎯</span>
                    <span class="sample-name">${file}</span>
                `;
                item.addEventListener("click", () => runSamplePrediction(file, item));
                sampleList.appendChild(item);
            });
        } catch (err) {
            console.error("Error fetching samples:", err);
            sampleList.innerHTML = `<div class="loading-spinner">Failed to load samples.</div>`;
        }
    }

    // Run prediction on a pre-loaded sample
    async function runSamplePrediction(filename, itemElement) {
        if (cameraPollingInterval) {
            disconnectCamera();
        }

        // Update UI state
        document.querySelectorAll(".sample-item").forEach(item => item.classList.remove("active"));
        if (itemElement) itemElement.classList.add("active");
        
        setLoadingState();
        resetZoomAndPan();

        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ sample_name: filename })
            });
            const data = await res.json();
            
            if (data.error) {
                showErrorState(data.error);
                return;
            }

            // Update Viewports
            originalPreview.src = `/api/sample/${filename}`;
            originalPreview.classList.remove("hidden");
            originalPlaceholder.classList.add("hidden");
            controlsOriginal.classList.remove("hidden");

            annotatedPreview.src = data.image_data;
            annotatedPreview.classList.remove("hidden");
            annotatedPlaceholder.classList.add("hidden");
            controlsAnnotated.classList.remove("hidden");

            updateTelemetry(data);
        } catch (err) {
            console.error("Error analyzing sample:", err);
            showErrorState("Server connection failed.");
        }
    }

    // Setup Drag-and-Drop & File Upload
    function setupUploadHandlers() {
        // Click to browse
        fileUpload.addEventListener("change", (e) => {
            const file = e.target.files[0];
            if (file) handleImageUpload(file);
        });

        // Drag events
        ["dragenter", "dragover"].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropZone.classList.add("dragover");
            }, false);
        });

        ["dragleave", "drop"].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropZone.classList.remove("dragover");
            }, false);
        });

        dropZone.addEventListener("drop", (e) => {
            const dt = e.dataTransfer;
            const file = dt.files[0];
            if (file) handleImageUpload(file);
        });
    }

    // Send uploaded image file to server
    async function handleImageUpload(file) {
        if (cameraPollingInterval) {
            disconnectCamera();
        }

        // Deselect any active samples
        document.querySelectorAll(".sample-item").forEach(item => item.classList.remove("active"));
        
        setLoadingState();
        resetZoomAndPan();

        // Show local preview of original image
        const reader = new FileReader();
        reader.onload = (e) => {
            originalPreview.src = e.target.result;
            originalPreview.classList.remove("hidden");
            originalPlaceholder.classList.add("hidden");
            controlsOriginal.classList.remove("hidden");
        };
        reader.readAsDataURL(file);

        // Send to backend
        const formData = new FormData();
        formData.append("file", file);

        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                body: formData
            });
            const data = await res.json();
            
            if (data.error) {
                showErrorState(data.error);
                return;
            }

            annotatedPreview.src = data.image_data;
            annotatedPreview.classList.remove("hidden");
            annotatedPlaceholder.classList.add("hidden");
            controlsAnnotated.classList.remove("hidden");

            updateTelemetry(data);
        } catch (err) {
            console.error("Error uploading image:", err);
            showErrorState("Server connection failed.");
        }
    }

    // Zoom & Pan Actions
    function updateZoomAndPan() {
        // Apply synchronized transform: translate then scale
        const transformStr = `translate(${panOffset.x}px, ${panOffset.y}px) scale(${zoomLevel})`;
        originalPreview.style.transform = transformStr;
        annotatedPreview.style.transform = transformStr;
    }

    function resetZoomAndPan() {
        zoomLevel = 1.0;
        panOffset = { x: 0, y: 0 };
        updateZoomAndPan();
    }

    function setupZoomAndPanHandlers() {
        // 1. Hook up toolbar buttons
        document.querySelectorAll(".zoom-in-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                zoomLevel = Math.min(5.0, zoomLevel + 0.25);
                updateZoomAndPan();
            });
        });

        document.querySelectorAll(".zoom-out-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                zoomLevel = Math.max(1.0, zoomLevel - 0.25);
                if (zoomLevel === 1.0) panOffset = { x: 0, y: 0 };
                updateZoomAndPan();
            });
        });

        document.querySelectorAll(".reset-btn").forEach(btn => {
            btn.addEventListener("click", resetZoomAndPan);
        });

        // 2. Mouse wheel zoom inside viewports
        const viewports = [
            document.getElementById("viewport-original"),
            document.getElementById("viewport-annotated")
        ];

        viewports.forEach(vp => {
            vp.addEventListener("wheel", (e) => {
                // Only zoom if an image is loaded
                if (originalPreview.classList.contains("hidden")) return;
                e.preventDefault();

                if (e.deltaY < 0) {
                    zoomLevel = Math.min(5.0, zoomLevel + 0.1);
                } else {
                    zoomLevel = Math.max(1.0, zoomLevel - 0.1);
                    if (zoomLevel === 1.0) panOffset = { x: 0, y: 0 };
                }
                updateZoomAndPan();
            }, { passive: false });
        });

        // 3. Mouse Drag Panning (bound to images)
        const images = [originalPreview, annotatedPreview];
        
        images.forEach(img => {
            img.addEventListener("mousedown", (e) => {
                if (zoomLevel <= 1.0) return; // Only pan when zoomed in
                e.preventDefault(); // Stop default image ghosting
                
                isDragging = true;
                dragStart.x = e.clientX - panOffset.x;
                dragStart.y = e.clientY - panOffset.y;
            });
        });

        window.addEventListener("mousemove", (e) => {
            if (!isDragging) return;
            panOffset.x = e.clientX - dragStart.x;
            panOffset.y = e.clientY - dragStart.y;
            updateZoomAndPan();
        });

        window.addEventListener("mouseup", () => {
            isDragging = false;
        });

        // Touch support for mobile panning
        images.forEach(img => {
            img.addEventListener("touchstart", (e) => {
                if (zoomLevel <= 1.0 || e.touches.length !== 1) return;
                isDragging = true;
                dragStart.x = e.touches[0].clientX - panOffset.x;
                dragStart.y = e.touches[0].clientY - panOffset.y;
            });

            img.addEventListener("touchmove", (e) => {
                if (!isDragging || e.touches.length !== 1) return;
                panOffset.x = e.touches[0].clientX - dragStart.x;
                panOffset.y = e.touches[0].clientY - dragStart.y;
                updateZoomAndPan();
            });
        });

        window.addEventListener("touchend", () => {
            isDragging = false;
        });
    }

    // Set Loading State
    function setLoadingState() {
        globalStatus.textContent = "Processing Feed...";
        globalStatus.className = "badge processing";
        
        verdictDisplay.textContent = "WAIT";
        verdictDisplay.className = "verdict idle";
        
        angleDisplay.textContent = "--°";
        gaugeNeedle.style.transform = "rotate(0deg)";
        
        controlsOriginal.classList.add("hidden");
        controlsAnnotated.classList.add("hidden");
    }

    // Show Error State
    function showErrorState(msg) {
        globalStatus.textContent = "Error";
        globalStatus.className = "badge ng";
        
        verdictDisplay.textContent = "FAIL";
        verdictDisplay.className = "verdict ng";
        
        angleDisplay.textContent = "Error";
        gaugeNeedle.style.transform = "rotate(0deg)";
        
        annotatedPreview.classList.add("hidden");
        annotatedPlaceholder.classList.remove("hidden");
        
        controlsOriginal.classList.add("hidden");
        controlsAnnotated.classList.add("hidden");
        
        detectionsLogBody.innerHTML = `<tr><td colspan="3" class="empty-table">${msg}</td></tr>`;
    }

    // Update Telemetry Panel
    function updateTelemetry(data) {
        // Map status to valid CSS class
        let cssClass = "idle";
        const statusUpper = data.status.toUpperCase();
        if (statusUpper.includes("OK") || statusUpper.includes("TRUE")) {
            cssClass = "ok";
        } else if (statusUpper.includes("NG") || statusUpper.includes("FALSE")) {
            cssClass = "ng";
        } else if (statusUpper.includes("WAIT") || statusUpper.includes("PROCESSING")) {
            cssClass = "processing";
        }

        // Update verdict badge
        globalStatus.textContent = `System: ${data.status}`;
        globalStatus.className = `badge ${cssClass}`;
        
        // Update Verdict Display Card
        verdictDisplay.textContent = data.status;
        verdictDisplay.className = `verdict ${cssClass}`;
        
        // Update Angle
        if (data.angle !== null) {
            angleDisplay.textContent = `${data.angle.toFixed(1)}°`;
            // Gauge rotation: calculated angle maps where 90deg is straight up (0 offset)
            // needle rotation offset = angle - 90
            const rotation = data.angle - 90;
            // Bound rotation to fit visual limits (-90 to +90 degrees)
            const boundedRotation = Math.max(-90, Math.min(90, rotation));
            gaugeNeedle.style.transform = `rotate(${boundedRotation}deg)`;
        } else {
            angleDisplay.textContent = "--°";
            gaugeNeedle.style.transform = "rotate(0deg)";
        }
        
        // Update Confidence Bars
        centerConfVal.textContent = `${Math.round(data.center_conf * 100)}%`;
        centerConfBar.style.width = `${Math.round(data.center_conf * 100)}%`;
        
        notchConfVal.textContent = `${Math.round(data.notch_conf * 100)}%`;
        notchConfBar.style.width = `${Math.round(data.notch_conf * 100)}%`;
        
        // Populate Detections Log Table
        detectionsLogBody.innerHTML = "";
        if (data.detections.length === 0) {
            detectionsLogBody.innerHTML = `<tr><td colspan="3" class="empty-table">No bounding boxes detected</td></tr>`;
            return;
        }
        
        data.detections.forEach(det => {
            const row = document.createElement("tr");
            const centerStr = `(${det.center[0]}, ${det.center[1]})`;
            const confStr = `${(det.confidence * 100).toFixed(0)}%`;
            
            row.innerHTML = `
                <td><span class="class-label ${det.class}">${det.class}</span></td>
                <td>${centerStr}</td>
                <td><strong>${confStr}</strong></td>
            `;
            detectionsLogBody.appendChild(row);
        });
    }

    function setupCameraHandlers() {
        connectBtn.addEventListener("click", connectCamera);
        disconnectBtn.addEventListener("click", disconnectCamera);
    }

    async function connectCamera() {
        const url = cameraUrlInput.value.trim();
        setLoadingState();
        resetZoomAndPan();

        try {
            const res = await fetch("/api/camera/connect", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ url: url })
            });
            const data = await res.json();

            if (!data.success) {
                showErrorState(data.message);
                return;
            }

            // Update button visibility
            connectBtn.classList.add("hidden");
            disconnectBtn.classList.remove("hidden");

            // Hide original viewport and show streaming state placeholder
            originalPreview.classList.add("hidden");
            originalPlaceholder.classList.remove("hidden");
            originalPlaceholder.querySelector("p").textContent = "Camera Streaming Active: Live processed feed shown on right viewport.";
            controlsOriginal.classList.add("hidden");

            // Set source of annotated preview to Flask MJPEG stream
            annotatedPreview.src = "/api/camera/stream";
            annotatedPreview.classList.remove("hidden");
            annotatedPlaceholder.classList.add("hidden");
            controlsAnnotated.classList.remove("hidden");

            // Start polling telemetry status
            if (cameraPollingInterval) clearInterval(cameraPollingInterval);
            cameraPollingInterval = setInterval(pollCameraStatus, 300);
        } catch (err) {
            console.error("Error connecting camera:", err);
            showErrorState("Failed to connect to camera.");
        }
    }

    async function disconnectCamera() {
        if (cameraPollingInterval) {
            clearInterval(cameraPollingInterval);
            cameraPollingInterval = null;
        }

        try {
            await fetch("/api/camera/disconnect", { method: "POST" });
        } catch (err) {
            console.error("Error disconnecting camera:", err);
        }

        connectBtn.classList.remove("hidden");
        disconnectBtn.classList.add("hidden");

        annotatedPreview.src = "";
        annotatedPreview.classList.add("hidden");
        annotatedPlaceholder.classList.remove("hidden");
        controlsAnnotated.classList.add("hidden");

        originalPlaceholder.querySelector("p").textContent = "Select a sample image or upload a custom raw image feed";
        globalStatus.textContent = "System Idle";
        globalStatus.className = "badge idle";
        verdictDisplay.textContent = "IDLE";
        verdictDisplay.className = "verdict idle";
        angleDisplay.textContent = "--°";
        gaugeNeedle.style.transform = "rotate(0deg)";
        centerConfVal.textContent = "0%";
        centerConfBar.style.width = "0%";
        notchConfVal.textContent = "0%";
        notchConfBar.style.width = "0%";
        detectionsLogBody.innerHTML = `<tr><td colspan="3" class="empty-table">No active detections</td></tr>`;
    }

    async function pollCameraStatus() {
        try {
            const res = await fetch("/api/camera/status");
            const data = await res.json();
            
            // Map status OK -> TRUE (OK), NG -> FALSE (NG)
            let displayData = { ...data };
            if (data.status === "OK") {
                displayData.status = "TRUE (OK)";
            } else if (data.status === "NG") {
                displayData.status = "FALSE (NG)";
            }
            updateTelemetry(displayData);
        } catch (err) {
            console.error("Error polling camera status:", err);
        }
    }
});
