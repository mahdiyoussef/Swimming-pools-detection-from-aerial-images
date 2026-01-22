import React, { useState, useEffect } from 'react';
import axios from 'axios';
import MapComponent from './components/MapComponent';
import Sidebar from './components/Sidebar';
import { Layers, Map as MapIcon, Info, Activity } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const App = () => {
    const [imageList, setImageList] = useState([]);
    const [selectedImage, setSelectedImage] = useState(null);
    const [geoData, setGeoData] = useState(null);
    const [liveData, setLiveData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [isScanning, setIsScanning] = useState(false);

    useEffect(() => {
        fetchImages();
    }, []);

    const fetchImages = async () => {
        try {
            const response = await axios.get('/api/detections');
            setImageList(response.data);
            if (response.data.length > 0) {
                handleSelectImage(response.data[0]);
            }
        } catch (error) {
            console.error('Error fetching images:', error);
        }
    };

    const handleSelectImage = async (name) => {
        setLoading(true);
        setSelectedImage(name);
        setLiveData(null); // Clear live data when switching images
        try {
            const response = await axios.get(`/api/detections/${name}/geojson`);
            setGeoData(response.data);
        } catch (error) {
            console.error('Error fetching geojson:', error);
            setGeoData(null);
        } finally {
            setLoading(false);
        }
    };

    const handleLiveDetect = async (bounds) => {
        setIsScanning(true);
        try {
            const response = await axios.post('/api/detect_live', bounds);
            // Merge or replace? For "Live" we probably want to append or just show new ones
            setLiveData(response.data);
        } catch (error) {
            console.error('Live detection failed:', error);
            alert('Failed to scan map. Make sure the backend is running with the model loaded.');
        } finally {
            setIsScanning(false);
        }
    };

    // Combine static and live data for total stats
    const combinedFeatures = [
        ...(geoData?.features || []),
        ...(liveData?.features || [])
    ];

    return (
        <div className="flex h-screen w-screen bg-gray-900 text-white overflow-hidden">
            {/* Sidebar */}
            <Sidebar
                imageList={imageList}
                selectedImage={selectedImage}
                onSelect={handleSelectImage}
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
                        <div className="flex items-center gap-2">
                            <Activity className="w-4 h-4 text-green-400" />
                            <span>Model: YOLOv11s</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <Layers className="w-4 h-4 text-pool" />
                            <span>Mode: {isScanning ? 'Scanning...' : 'Geospatial Tiling'}</span>
                        </div>
                    </div>
                </header>

                {/* Map Area */}
                <div className="flex-1 relative bg-gray-950">
                    <AnimatePresence mode="wait">
                        {!loading && geoData ? (
                            <motion.div
                                key={selectedImage}
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                                className="w-full h-full"
                            >
                                <MapComponent
                                    data={{ ...geoData, features: combinedFeatures }}
                                    imageName={selectedImage}
                                    onLiveDetect={handleLiveDetect}
                                    isScanning={isScanning}
                                />
                            </motion.div>
                        ) : (
                            <div className="absolute inset-0 flex items-center justify-center">
                                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-pool"></div>
                            </div>
                        )}
                    </AnimatePresence>
                </div>

                {/* Floating Stats */}
                {geoData && !loading && (
                    <motion.div
                        initial={{ y: 20, opacity: 0 }}
                        animate={{ y: 0, opacity: 1 }}
                        className="absolute bottom-6 right-6 p-4 bg-gray-900/80 backdrop-blur-xl border border-gray-700 rounded-2xl shadow-2xl z-20 min-w-[200px]"
                    >
                        <div className="flex flex-col gap-3">
                            <div className="flex justify-between items-center">
                                <span className="text-gray-400 text-xs uppercase tracking-wider">Total Pools</span>
                                <span className="text-xl font-bold text-pool">{combinedFeatures.length}</span>
                            </div>
                            <div className="h-px bg-gray-800" />
                            <div className="flex justify-between items-center">
                                <span className="text-gray-400 text-xs uppercase tracking-wider">Estimated Area</span>
                                <span className="text-xl font-bold text-white">
                                    {combinedFeatures.reduce((acc, f) => acc + f.properties.area_m2, 0).toFixed(1)} m²
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
