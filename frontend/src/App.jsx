import React, { useState } from 'react';
import axios from 'axios';
import MapComponent from './components/MapComponent';
import Sidebar from './components/Sidebar';
import { Layers, Map as MapIcon, Activity, Download } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const App = () => {
    const [geoData, setGeoData] = useState(null);
    const [isScanning, setIsScanning] = useState(false);

    // Store all live scans with timestamps
    const [liveScans, setLiveScans] = useState([]);
    const [selectedLiveScan, setSelectedLiveScan] = useState(null);

    const handleSelectLiveScan = (scan) => {
        setSelectedLiveScan(scan.id);
        setGeoData(scan.data);
    };

    const handleLiveDetect = async (bounds) => {
        setIsScanning(true);
        try {
            const response = await axios.post('/api/detect_live', bounds);
            const scanData = response.data;

            // Create a new live scan entry
            const newScan = {
                id: `live_${Date.now()}`,
                name: `Scan ${liveScans.length + 1}`,
                timestamp: new Date().toISOString(),
                bounds: bounds,
                data: scanData,
                poolCount: scanData.features?.length || 0
            };

            // Add to live scans list
            setLiveScans(prev => [newScan, ...prev]);

            // Show this scan
            setGeoData(scanData);
            setSelectedLiveScan(newScan.id);

        } catch (error) {
            console.error('Live detection failed:', error);
            alert('Failed to scan map. Make sure the backend is running with the model loaded.');
        } finally {
            setIsScanning(false);
        }
    };

    // Export functions
    const exportAsJSON = () => {
        if (!geoData) return;

        const blob = new Blob([JSON.stringify(geoData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `pool_detections_${selectedLiveScan || 'export'}.geojson`;
        a.click();
        URL.revokeObjectURL(url);
    };

    const exportAsCSV = () => {
        if (!geoData || !geoData.features) return;

        // CSV headers
        const headers = ['id', 'latitude', 'longitude', 'confidence', 'shape', 'area_m2'];

        // Convert features to CSV rows
        const rows = geoData.features.map((feature, idx) => {
            const coords = feature.geometry.coordinates[0];
            // Get center point (average of all coordinates)
            const centerLat = coords.reduce((sum, c) => sum + c[1], 0) / coords.length;
            const centerLng = coords.reduce((sum, c) => sum + c[0], 0) / coords.length;

            return [
                feature.id || `pool_${idx}`,
                centerLat.toFixed(6),
                centerLng.toFixed(6),
                (feature.properties.confidence * 100).toFixed(1) + '%',
                feature.properties.shape,
                feature.properties.area_m2
            ].join(',');
        });

        const csv = [headers.join(','), ...rows].join('\n');
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `pool_detections_${selectedLiveScan || 'export'}.csv`;
        a.click();
        URL.revokeObjectURL(url);
    };

    const features = geoData?.features || [];

    return (
        <div className="flex h-screen w-screen bg-gray-900 text-white overflow-hidden">
            {/* Sidebar */}
            <Sidebar
                liveScans={liveScans}
                selectedLiveScan={selectedLiveScan}
                onSelectLiveScan={handleSelectLiveScan}
            />

            {/* Main Content */}
            <main className="flex-1 relative flex flex-col">
                {/* Header */}
                <header className="h-16 border-b border-gray-800 flex items-center justify-between px-6 bg-gray-900/50 backdrop-blur-md z-10">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-pool rounded-lg">
                            <MapIcon className="w-5 h-5 text-gray-900" />
                        </div>
                        <h1 className="text-xl font-bold tracking-tight">
                            Pool Detection <span className="text-pool font-normal">Dashboard</span>
                        </h1>
                    </div>

                    <div className="flex items-center gap-6 text-sm text-gray-400">
                        {/* Export Buttons */}
                        {geoData && (
                            <div className="flex items-center gap-2">
                                <button
                                    onClick={exportAsJSON}
                                    className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-800 hover:bg-gray-700 rounded-lg text-xs font-medium transition-colors"
                                >
                                    <Download className="w-3 h-3" />
                                    GeoJSON
                                </button>
                                <button
                                    onClick={exportAsCSV}
                                    className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-800 hover:bg-gray-700 rounded-lg text-xs font-medium transition-colors"
                                >
                                    <Download className="w-3 h-3" />
                                    CSV
                                </button>
                            </div>
                        )}

                        <div className="flex items-center gap-2">
                            <Activity className="w-4 h-4 text-green-400" />
                            <span>Model: YOLOv11s</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <Layers className="w-4 h-4 text-pool" />
                            <span>Mode: {isScanning ? 'Scanning...' : 'Sliding Window'}</span>
                        </div>
                    </div>
                </header>

                {/* Map Area */}
                <div className="flex-1 relative bg-gray-950">
                    <MapComponent
                        data={geoData}
                        onLiveDetect={handleLiveDetect}
                        isScanning={isScanning}
                    />
                </div>

                {/* Floating Stats */}
                {geoData && features.length > 0 && (
                    <motion.div
                        initial={{ y: 20, opacity: 0 }}
                        animate={{ y: 0, opacity: 1 }}
                        className="absolute bottom-6 right-6 p-4 bg-gray-900/80 backdrop-blur-xl border border-gray-700 rounded-2xl shadow-2xl z-20 min-w-[200px]"
                    >
                        <div className="flex flex-col gap-3">
                            <div className="flex justify-between items-center">
                                <span className="text-gray-400 text-xs uppercase tracking-wider">Total Pools</span>
                                <span className="text-xl font-bold text-pool">{features.length}</span>
                            </div>
                            <div className="h-px bg-gray-800" />
                            <div className="flex justify-between items-center">
                                <span className="text-gray-400 text-xs uppercase tracking-wider">Estimated Area</span>
                                <span className="text-xl font-bold text-white">
                                    {features.reduce((acc, f) => acc + (f.properties?.area_m2 || 0), 0).toFixed(1)} m²
                                </span>
                            </div>
                        </div>
                    </motion.div>
                )}
            </main>
        </div>
    );
};

export default App;
