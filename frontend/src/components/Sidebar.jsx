import React from 'react';
import { Search, Scan, Clock, ChevronRight, MapPin } from 'lucide-react';

const Sidebar = ({
    liveScans = [],
    selectedLiveScan,
    onSelectLiveScan
}) => {
    return (
        <aside className="w-80 h-full border-r border-gray-800 bg-gray-900 flex flex-col z-30">
            <div className="p-6">
                <div className="relative">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
                    <input
                        type="text"
                        placeholder="Search scans..."
                        className="w-full bg-gray-800 border-none rounded-xl py-2.5 pl-10 pr-4 text-sm focus:ring-2 focus:ring-pool transition-all"
                    />
                </div>
            </div>

            <div className="flex-1 overflow-y-auto px-4 pb-6">
                {/* Live Scans Section */}
                <h2 className="px-2 text-xs font-semibold text-green-400 uppercase tracking-widest mb-4 flex items-center gap-2">
                    <Scan className="w-3 h-3" />
                    Live Scans {liveScans.length > 0 && `(${liveScans.length})`}
                </h2>

                <div className="space-y-1">
                    {liveScans.map((scan) => (
                        <button
                            key={scan.id}
                            onClick={() => onSelectLiveScan(scan)}
                            className={`w-full flex items-center gap-3 px-3 py-3 rounded-2xl transition-all group ${selectedLiveScan === scan.id
                                    ? 'bg-green-500/10 text-green-400 shadow-inner'
                                    : 'hover:bg-gray-800 text-gray-400'
                                }`}
                        >
                            <div className={`p-2 rounded-lg ${selectedLiveScan === scan.id
                                    ? 'bg-green-500/20'
                                    : 'bg-gray-800 group-hover:bg-gray-700'
                                }`}>
                                <MapPin className="w-4 h-4" />
                            </div>
                            <div className="flex-1 text-left overflow-hidden">
                                <p className="text-sm font-medium truncate">{scan.name}</p>
                                <div className="flex items-center gap-2 text-[10px] opacity-60">
                                    <span className="flex items-center gap-1">
                                        <Clock className="w-2.5 h-2.5" />
                                        {new Date(scan.timestamp).toLocaleTimeString()}
                                    </span>
                                    <span>•</span>
                                    <span className="text-green-400 font-medium">{scan.poolCount} pools</span>
                                </div>
                            </div>
                            <ChevronRight className={`w-4 h-4 transition-transform ${selectedLiveScan === scan.id
                                    ? 'translate-x-0'
                                    : '-translate-x-2 opacity-0 group-hover:translate-x-0 group-hover:opacity-100'
                                }`} />
                        </button>
                    ))}

                    {liveScans.length === 0 && (
                        <div className="py-16 text-center">
                            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-gray-800 flex items-center justify-center">
                                <Scan className="w-8 h-8 text-gray-600" />
                            </div>
                            <p className="text-gray-500 text-sm font-medium mb-1">No scans yet</p>
                            <p className="text-gray-600 text-xs">
                                Click "Scan This Area" on the map<br />to detect swimming pools
                            </p>
                        </div>
                    )}
                </div>
            </div>

            <div className="p-6 border-t border-gray-800">
                <div className="bg-gray-950/50 rounded-2xl p-4 border border-gray-800">
                    <p className="text-[10px] text-gray-500 font-bold uppercase mb-2">System Status</p>
                    <div className="flex items-center gap-2">
                        <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                        <span className="text-xs font-medium text-gray-300">Backend API Online</span>
                    </div>
                </div>
            </div>
        </aside>
    );
};

export default Sidebar;
