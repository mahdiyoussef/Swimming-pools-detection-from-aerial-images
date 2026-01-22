import React, { useState, useRef } from 'react';
import { MapContainer, TileLayer, Polygon, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import { Activity } from 'lucide-react';

// Fix for Leaflet marker icons in React/Vite
// Using standard CDN URLs for default markers
const DefaultIcon = L.icon({
    iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
    shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
    iconSize: [25, 41],
    iconAnchor: [12, 41]
});

L.Marker.prototype.options.icon = DefaultIcon;

// Default center: Alpes-Maritimes region (nice area with many pools)
const DEFAULT_CENTER = [43.7, 7.2];
const DEFAULT_ZOOM = 18;

const MapComponent = ({ data, onLiveDetect, isScanning }) => {
    // Inner component to handle map events and button
    const ControlComponent = () => {
        const map = useMap();

        const handleScan = () => {
            const bounds = map.getBounds();
            const zoom = map.getZoom();

            const payload = {
                north: bounds.getNorth(),
                south: bounds.getSouth(),
                east: bounds.getEast(),
                west: bounds.getWest(),
                zoom: Math.round(zoom)
            };

            onLiveDetect(payload);
        };

        return (
            <div className="absolute top-6 left-6 z-[1000] flex flex-col gap-2">
                <button
                    onClick={handleScan}
                    disabled={isScanning}
                    className={`flex items-center gap-2 px-6 py-3 rounded-2xl font-bold shadow-2xl transition-all ${isScanning
                        ? 'bg-gray-700 text-gray-500 cursor-not-allowed scale-95'
                        : 'bg-pool text-gray-900 hover:scale-105 active:scale-95'
                        }`}
                >
                    {isScanning ? (
                        <>
                            <div className="w-4 h-4 border-2 border-gray-900/30 border-t-gray-900 rounded-full animate-spin" />
                            Scanning Viewport...
                        </>
                    ) : (
                        <>
                            <Activity className="w-5 h-5 transition-transform group-hover:rotate-12" />
                            Scan This Area
                        </>
                    )}
                </button>
            </div>
        );
    };

    return (
        <MapContainer
            center={DEFAULT_CENTER}
            zoom={DEFAULT_ZOOM}
            scrollWheelZoom={true}
            zoomControl={false}
            fadeAnimation={true}
            style={{ background: '#0f172a' }}
        >
            {/* Esri High-Resolution Satellite Tiles */}
            <TileLayer
                attribution='&copy; <a href="https://www.esri.com/">Esri</a>'
                url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                maxZoom={19}
                updateWhenIdle={true}
                keepBuffer={4}
            />

            <ControlComponent />

            {data && data.features && data.features.map((feature, idx) => (
                <Polygon
                    key={`${feature.id || idx}`}
                    positions={feature.geometry.coordinates[0].map(coord => [coord[1], coord[0]])}
                    pathOptions={{
                        color: feature.properties.is_live ? '#10b981' : '#ef4444', // Green for live, Red for saved
                        fillColor: '#4fd1c5',
                        fillOpacity: 0.35,
                        weight: 3,
                        lineJoin: 'round'
                    }}
                >
                    <Popup className="custom-popup">
                        <div className="p-2 min-w-[120px]">
                            <h3 className="text-sm font-bold text-gray-900 mb-1 capitalize border-b pb-1">
                                {feature.properties.shape} Pool
                            </h3>
                            <div className="text-xs text-gray-700 flex flex-col gap-1 mt-2">
                                <p className="flex justify-between">
                                    <span>Area:</span>
                                    <span className="font-bold text-pool-dark">{feature.properties.area_m2} m²</span>
                                </p>
                                <p className="flex justify-between">
                                    <span>Confidence:</span>
                                    <span className="font-bold">{(feature.properties.confidence * 100).toFixed(1)}%</span>
                                </p>
                            </div>
                        </div>
                    </Popup>
                </Polygon>
            ))}
        </MapContainer>
    );
};

export default MapComponent;
