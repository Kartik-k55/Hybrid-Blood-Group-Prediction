/**
 * Rakta Mitra - Interactive 3D Hero Visuals
 * Realistic 3D Fingerprint Biometrics + Floating Biconcave Red Blood Cells (Erythrocytes)
 * Powered by Three.js with Smooth Parallax, Biometric Scanning Ring & Particle Nodes
 */

(function () {
    'use strict';

    // Check prefers-reduced-motion
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // Check WebGL availability
    function isWebGLAvailable() {
        try {
            const canvas = document.createElement('canvas');
            return !!(window.WebGLRenderingContext && (canvas.getContext('webgl') || canvas.getContext('experimental-webgl')));
        } catch (e) {
            return false;
        }
    }

    const container = document.getElementById('hero-3d-canvas-container');
    if (!container || !window.THREE || !isWebGLAvailable()) {
        if (container) {
            container.classList.add('fallback-mode');
        }
        return;
    }

    // 1. Scene, Camera, Renderer
    const width = container.clientWidth || 540;
    const height = container.clientHeight || 480;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(40, width / height, 0.1, 100);
    camera.position.set(0, 0, 9);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    // 2. Lighting Setup
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
    scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0xffffff, 0.95);
    keyLight.position.set(5, 8, 7);
    scene.add(keyLight);

    const sageFillLight = new THREE.DirectionalLight(0xd6e4d7, 0.6);
    sageFillLight.position.set(-6, -4, 5);
    scene.add(sageFillLight);

    const softCrimsonLight = new THREE.PointLight(0xfca5a5, 0.5, 15);
    softCrimsonLight.position.set(3, -2, 4);
    scene.add(softCrimsonLight);

    // Master Group for Mouse Parallax
    const heroGroup = new THREE.Group();
    scene.add(heroGroup);

    // 3. Biometric Fingerprint Construction
    const fingerprintGroup = new THREE.Group();
    heroGroup.add(fingerprintGroup);

    // Generate Anatomical Fingerprint Ridges (Concentric Whorls & Loops)
    const ridgeMaterial = new THREE.MeshStandardMaterial({
        color: 0x294936, // Dark Forest Green
        roughness: 0.35,
        metalness: 0.2,
        flatShading: false
    });

    const numRidges = 24;
    const ridgeCurves = [];

    for (let i = 1; i <= numRidges; i++) {
        const rNorm = i / numRidges;
        const rx = 0.35 + rNorm * 1.55;
        const ry = 0.5 + rNorm * 2.15;
        const points = [];
        const steps = 72;

        // Loop / Whorl with subtle biometric minutiae undulations
        for (let j = 0; j <= steps; j++) {
            const theta = (j / steps) * Math.PI * 2;
            // Elliptical arch with upward core pull
            const yOffset = -0.15 * Math.sin(theta);
            // Ridge wave harmonics for organic realism
            const wave = 0.025 * Math.sin(theta * 6 + i * 0.7);
            const x = (rx + wave) * Math.sin(theta);
            const y = (ry + wave + yOffset) * Math.cos(theta);
            // 3D convex dome profile (finger pad curve)
            const z = Math.max(0, (1 - (x * x) / (2.2 * 2.2) - (y * y) / (2.8 * 2.8))) * 0.42;

            points.push(new THREE.Vector3(x, y, z));
        }

        const curve = new THREE.CatmullRomCurve3(points, true);
        ridgeCurves.push(curve);

        const tubeGeom = new THREE.TubeGeometry(curve, 64, 0.022, 6, true);
        const tubeMesh = new THREE.Mesh(tubeGeom, ridgeMaterial);
        fingerprintGroup.add(tubeMesh);
    }

    // Circular Biometric Scanning Ring with Angle Markings
    const ringGroup = new THREE.Group();
    fingerprintGroup.add(ringGroup);

    const ringGeom = new THREE.RingGeometry(2.7, 2.74, 64);
    const ringMat = new THREE.MeshBasicMaterial({
        color: 0x888888,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.45
    });
    const ringMesh = new THREE.Mesh(ringGeom, ringMat);
    ringGroup.add(ringMesh);

    // Dashed Outer Calibration Ring
    const outerRingGeom = new THREE.RingGeometry(2.9, 2.92, 48);
    const outerRingMat = new THREE.MeshBasicMaterial({
        color: 0x294936,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.35
    });
    const outerRingMesh = new THREE.Mesh(outerRingGeom, outerRingMat);
    ringGroup.add(outerRingMesh);

    // Rotating Biometric Tick Marks
    const ticksGroup = new THREE.Group();
    ringGroup.add(ticksGroup);
    const tickGeom = new THREE.BoxGeometry(0.04, 0.12, 0.02);
    const tickMat = new THREE.MeshBasicMaterial({ color: 0x294936, transparent: true, opacity: 0.6 });
    for (let k = 0; k < 12; k++) {
        const angle = (k / 12) * Math.PI * 2;
        const tick = new THREE.Mesh(tickGeom, tickMat);
        tick.position.set(Math.cos(angle) * 2.82, Math.sin(angle) * 2.82, 0.01);
        tick.rotation.z = angle;
        ticksGroup.add(tick);
    }

    // Glowing Green Scanning Nodes traveling along fingerprint ridges
    const scanNodes = [];
    const nodeGeom = new THREE.SphereGeometry(0.048, 12, 12);
    const nodeMat = new THREE.MeshBasicMaterial({ color: 0x22c55e });

    for (let n = 0; n < 8; n++) {
        const node = new THREE.Mesh(nodeGeom, nodeMat);
        const glowGeom = new THREE.SphereGeometry(0.09, 8, 8);
        const glowMat = new THREE.MeshBasicMaterial({ color: 0x86efac, transparent: true, opacity: 0.45 });
        const glowMesh = new THREE.Mesh(glowGeom, glowMat);
        node.add(glowMesh);

        fingerprintGroup.add(node);
        scanNodes.push({
            mesh: node,
            curveIndex: (n * 3) % ridgeCurves.length,
            speed: 0.08 + (n % 4) * 0.025,
            offset: n / 8
        });
    }

    // Vertical Laser Scanning Beam
    const laserBeamGeom = new THREE.PlaneGeometry(5.2, 0.04);
    const laserBeamMat = new THREE.MeshBasicMaterial({
        color: 0x294936,
        transparent: true,
        opacity: 0.75,
        side: THREE.DoubleSide
    });
    const laserBeam = new THREE.Mesh(laserBeamGeom, laserBeamMat);
    laserBeam.position.z = 0.48;
    fingerprintGroup.add(laserBeam);

    // Subtle Soft Ground Shadow Disc
    const shadowGeom = new THREE.CircleGeometry(2.6, 32);
    const shadowMat = new THREE.MeshBasicMaterial({
        color: 0x294936,
        transparent: true,
        opacity: 0.07,
        side: THREE.DoubleSide
    });
    const shadowMesh = new THREE.Mesh(shadowGeom, shadowMat);
    shadowMesh.position.set(0, -2.5, -0.6);
    shadowMesh.rotation.x = -Math.PI * 0.45;
    heroGroup.add(shadowMesh);

    // 4. Floating 3D Red Blood Cells (Biconcave Erythrocyte Discs)
    function createErythrocyteGeometry() {
        const points = [];
        const numPts = 28;
        // Top biconcave surface
        for (let i = 0; i <= numPts; i++) {
            const t = i / numPts;
            const r = t * 1.15;
            // Biconcave erythrocyte cross-section equation
            const y = 0.4 * Math.sqrt(Math.max(0, 1 - t * t)) * (0.35 + 2.1 * t * t - 1.05 * t * t * t * t);
            points.push(new THREE.Vector2(r, Math.max(0.04, y)));
        }
        // Bottom biconcave surface
        for (let i = numPts; i >= 0; i--) {
            const t = i / numPts;
            const r = t * 1.15;
            const y = -0.4 * Math.sqrt(Math.max(0, 1 - t * t)) * (0.35 + 2.1 * t * t - 1.05 * t * t * t * t);
            points.push(new THREE.Vector2(r, Math.min(-0.04, y)));
        }
        return new THREE.LatheGeometry(points, 32);
    }

    const rbcGeometry = createErythrocyteGeometry();
    const rbcMaterial = new THREE.MeshStandardMaterial({
        color: 0xb83b3b, // Soft muted medical red / crimson
        roughness: 0.32,
        metalness: 0.08,
        flatShading: false
    });

    const bloodCells = [];
    const cellConfigs = [
        { x: -2.8, y: 1.6, z: 0.8, scale: 0.55, rotSpeed: { x: 0.008, y: 0.012 }, floatSpeed: 1.2, floatAmp: 0.15 },
        { x: 3.1, y: 1.8, z: -0.5, scale: 0.72, rotSpeed: { x: -0.01, y: 0.007 }, floatSpeed: 0.9, floatAmp: 0.22 },
        { x: -3.2, y: -1.2, z: -0.8, scale: 0.65, rotSpeed: { x: 0.006, y: -0.009 }, floatSpeed: 1.1, floatAmp: 0.18 },
        { x: 2.7, y: -1.7, z: 0.6, scale: 0.58, rotSpeed: { x: 0.011, y: 0.008 }, floatSpeed: 1.3, floatAmp: 0.14 },
        { x: 1.8, y: 2.5, z: -1.2, scale: 0.48, rotSpeed: { x: -0.007, y: -0.012 }, floatSpeed: 1.0, floatAmp: 0.2 },
        { x: -1.9, y: -2.3, z: -0.4, scale: 0.52, rotSpeed: { x: 0.009, y: 0.01 }, floatSpeed: 1.4, floatAmp: 0.16 }
    ];

    cellConfigs.forEach(cfg => {
        const cell = new THREE.Mesh(rbcGeometry, rbcMaterial);
        cell.scale.set(cfg.scale, cfg.scale, cfg.scale);
        cell.position.set(cfg.x, cfg.y, cfg.z);
        cell.rotation.set(Math.random() * Math.PI, Math.random() * Math.PI, Math.random() * Math.PI);
        heroGroup.add(cell);

        bloodCells.push({
            mesh: cell,
            baseX: cfg.x,
            baseY: cfg.y,
            baseZ: cfg.z,
            rotSpeed: cfg.rotSpeed,
            floatSpeed: cfg.floatSpeed,
            floatAmp: cfg.floatAmp,
            seed: Math.random() * 10
        });
    });

    // 5. Interactive Mouse Parallax
    let mouseX = 0;
    let mouseY = 0;
    let targetRotX = 0;
    let targetRotY = 0;

    window.addEventListener('mousemove', (e) => {
        const rect = container.getBoundingClientRect();
        if (e.clientY >= rect.top - 200 && e.clientY <= rect.bottom + 200) {
            const normX = ((e.clientX - rect.left) / rect.width) * 2 - 1;
            const normY = -(((e.clientY - rect.top) / rect.height) * 2 - 1);
            targetRotY = normX * 0.35;
            targetRotX = -normY * 0.25;
        }
    });

    // 6. Responsive Resize Handling
    function handleResize() {
        const newW = container.clientWidth || 540;
        const newH = container.clientHeight || 480;
        camera.aspect = newW / newH;
        camera.updateProjectionMatrix();
        renderer.setSize(newW, newH);
    }
    window.addEventListener('resize', handleResize);

    // 7. Viewport Observer for CPU/Battery Optimization
    let isVisible = true;
    if ('IntersectionObserver' in window) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                isVisible = entry.isIntersecting;
            });
        }, { threshold: 0.1 });
        observer.observe(container);
    }

    // 8. Animation Loop
    let clock = new THREE.Clock();

    function animate() {
        requestAnimationFrame(animate);

        if (!isVisible || prefersReducedMotion) {
            renderer.render(scene, camera);
            return;
        }

        const elapsedTime = clock.getElapsedTime();

        // Smooth Mouse Parallax Damping
        heroGroup.rotation.y += (targetRotY - heroGroup.rotation.y) * 0.05;
        heroGroup.rotation.x += (targetRotX - heroGroup.rotation.x) * 0.05;

        // Gentle Floating and Slow Rotation of Fingerprint
        fingerprintGroup.position.y = Math.sin(elapsedTime * 0.8) * 0.08;
        fingerprintGroup.rotation.z = Math.sin(elapsedTime * 0.5) * 0.03;

        // Slow Rotation of Calibration Rings
        ringGroup.rotation.z = elapsedTime * 0.08;
        ticksGroup.rotation.z = -elapsedTime * 0.04;

        // Vertical Laser Scan Beam Movement
        laserBeam.position.y = Math.sin(elapsedTime * 1.5) * 2.2;

        // Animate Scanning Nodes along Ridge Curves
        scanNodes.forEach(node => {
            const curve = ridgeCurves[node.curveIndex];
            if (curve) {
                const u = (elapsedTime * node.speed + node.offset) % 1.0;
                const pos = curve.getPointAt(u);
                node.mesh.position.copy(pos);
            }
        });

        // Animate Floating Red Blood Cells
        bloodCells.forEach(cell => {
            cell.mesh.rotation.x += cell.rotSpeed.x;
            cell.mesh.rotation.y += cell.rotSpeed.y;
            cell.mesh.position.y = cell.baseY + Math.sin(elapsedTime * cell.floatSpeed + cell.seed) * cell.floatAmp;
            cell.mesh.position.x = cell.baseX + Math.cos(elapsedTime * (cell.floatSpeed * 0.7) + cell.seed) * (cell.floatAmp * 0.5);
        });

        renderer.render(scene, camera);
    }

    animate();
})();
