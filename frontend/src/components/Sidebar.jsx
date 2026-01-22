import React from 'react';
import { Search, Image as ImageIcon, ChevronRight } from 'lucide-react';

const Sidebar = ({ imageList, selectedImage, onSelect }) => {
    return (
        <aside className="w-80 h-full border-r border-gray-800 bg-gray-900 flex flex-col z-30">
            <div className="p-6">
                <div className="relative">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
                    <input
                        type="text"
                        placeholder="Search detections..."
                        className="w-full bg-gray-800 border-none rounded-xl py-2.5 pl-10 pr-4 text-sm focus:ring-2 focus:ring-pool transition-all"
                    />
                </div>
            </div>

            <div className="flex-1 overflow-y-auto px-4 pb-6 space-y-1">
                <h2 className="px-2 text-xs font-semibold text-gray-500 uppercase tracking-widest mb-4">
                    Processed Sequences
                </h2>

                {imageList.map((name) => (
                    <button
                        key={name}
                        onClick={() => onSelect(name)}
                        className={`w-full flex items-center gap-3 px-3 py-4 rounded-2xl transition-all group ${selectedImage === name
                                ? 'bg-pool/10 text-pool shadow-inner'
                                : 'hover:bg-gray-800 text-gray-400'
                            }`}
                    >
                        <div className={`p-2 rounded-lg ${selectedImage === name ? 'bg-pool/20' : 'bg-gray-800 group-hover:bg-gray-700'
                            }`}>
                            <ImageIcon className="w-4 h-4" />
                        </div>
                        <div className="flex-1 text-left overflow-hidden">
                            <p className="text-sm font-medium truncate">{name}</p>
                            <p className="text-[10px] opacity-60">Verified Shape Mask</p>
                        </div>
                        <ChevronRight className={`w-4 h-4 transition-transform ${selectedImage === name ? 'translate-x-0' : '-translate-x-2 opacity-0 group-hover:translate-x-0 group-hover:opacity-100'
                            }`} />
                    </button>
                ))}

                {imageList.length === 0 && (
                    <div className="py-12 text-center">
                        <p className="text-gray-600 text-sm italic">No detections found in output folder.</p>
                    </div>
                )}
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
