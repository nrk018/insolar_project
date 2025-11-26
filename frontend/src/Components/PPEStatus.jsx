import React from 'react';

const PPEStatus = ({ ppeCompliant, ppeItems, ppeItemsAccuracy, showDetails = false }) => {
  // Default values if not provided
  const isCompliant = ppeCompliant !== undefined ? ppeCompliant : false;
  const items = ppeItems || {};
  
  // PPE items to check
  const ppeItemList = [
    { key: 'helmet', label: 'Helmet', icon: '🪖' },
    { key: 'gloves', label: 'Gloves', icon: '🧤' },
    { key: 'boots', label: 'Boots', icon: '👢' },
    { key: 'jacket', label: 'Jacket', icon: '🧥' },
  ];

  // If no PPE data available
  if (ppeCompliant === undefined && (!ppeItems || Object.keys(ppeItems).length === 0)) {
    return (
      <span className="text-gray-400 text-xs">N/A</span>
    );
  }

  // Simple badge view (default)
  if (!showDetails) {
    return (
      <div className="flex items-center gap-1">
        <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-semibold ${
          isCompliant 
            ? 'bg-green-100 text-green-800 border border-green-200' 
            : 'bg-red-100 text-red-800 border border-red-200'
        }`}>
          {isCompliant ? '✓ PPE' : '✗ PPE'}
        </span>
      </div>
    );
  }

  // Detailed view with individual items
  return (
    <div className="flex flex-col gap-1">
      <div className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-semibold w-fit ${
        isCompliant 
          ? 'bg-green-100 text-green-800 border border-green-200' 
          : 'bg-red-100 text-red-800 border border-red-200'
      }`}>
        {isCompliant ? '✓ PPE Compliant' : '✗ PPE Non-Compliant'}
      </div>
      <div className="flex flex-wrap gap-2 text-xs">
        {ppeItemList.map((item) => {
          const detected = items[item.key] === true;
          const accuracy = ppeItemsAccuracy && ppeItemsAccuracy[item.key] !== undefined 
            ? (ppeItemsAccuracy[item.key] * 100).toFixed(1) 
            : null;
          return (
            <span
              key={item.key}
              className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded ${
                detected
                  ? 'bg-green-50 text-green-700'
                  : 'bg-red-50 text-red-700'
              }`}
              title={`${item.label}: ${detected ? 'Detected' : 'Missing'}${accuracy ? ` (${accuracy}% accuracy)` : ''}`}
            >
              <span>{item.icon}</span>
              <span>{item.label}</span>
              {detected ? <span className="ml-1">✓</span> : <span className="ml-1">✗</span>}
              {accuracy && <span className="ml-1 text-xs opacity-75">({accuracy}%)</span>}
            </span>
          );
        })}
      </div>
    </div>
  );
};

export default PPEStatus;

