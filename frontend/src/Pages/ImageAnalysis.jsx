import { useState } from 'react';
import flaskAxios from '../utils/flaskAxios';
import PPEStatus from '../Components/PPEStatus';

const ImageAnalysis = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const FLASK_URL = import.meta.env.VITE_FLASK_URL || 'http://localhost:5000';

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setError(null);
      setResults(null);
      
      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const formData = new FormData();
      formData.append('image', selectedFile);

      const response = await flaskAxios.post(`${FLASK_URL}/api/analyze-image`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setResults(response.data);
    } catch (err) {
      console.error('Error analyzing image:', err);
      const errorMessage = err.response?.data?.error || 
                          err.message || 
                          'Failed to analyze image. Make sure Flask server is running on port 5000.';
      setError(errorMessage);
      
      // Log full error for debugging
      if (err.response) {
        console.error('Response error:', err.response.data);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setSelectedFile(null);
    setPreview(null);
    setResults(null);
    setError(null);
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex flex-col space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">Image Analysis</h1>
        <p className="text-muted-foreground">
          Upload an image to identify employees and detect PPE compliance
        </p>
      </div>

      {/* Upload Section */}
      <div className="rounded-lg border bg-card p-6 space-y-4">
        <div>
          <label className="block text-sm font-medium mb-2">
            Select Image
          </label>
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
          />
        </div>

        {preview && (
          <div className="mt-4 space-y-4">
            <div>
              <p className="text-sm font-medium mb-2">Original Image:</p>
              <img
                src={preview}
                alt="Preview"
                className="max-w-full h-auto rounded-lg border"
                style={{ maxHeight: '400px' }}
              />
            </div>
            {results && results.annotated_image && (
              <div>
                <p className="text-sm font-medium mb-2">Annotated Image (with PPE detection boxes):</p>
                <img
                  src={results.annotated_image}
                  alt="Annotated Analysis"
                  className="max-w-full h-auto rounded-lg border"
                  style={{ maxHeight: '400px' }}
                />
              </div>
            )}
          </div>
        )}

        <div className="flex gap-4">
          <button
            onClick={handleAnalyze}
            disabled={!selectedFile || loading}
            className={`px-6 py-2 rounded-md font-medium ${
              !selectedFile || loading
                ? 'bg-gray-400 text-white cursor-not-allowed'
                : 'bg-blue-500 text-white hover:bg-blue-600'
            }`}
          >
            {loading ? 'Analyzing...' : 'Analyze Image'}
          </button>

          {(preview || results) && (
            <button
              onClick={handleClear}
              className="px-6 py-2 rounded-md font-medium bg-gray-500 text-white hover:bg-gray-600"
            >
              Clear
            </button>
          )}
        </div>

        {error && (
          <div className="rounded-md bg-red-50 border border-red-200 p-3">
            <p className="text-sm text-red-800">{error}</p>
          </div>
        )}
      </div>

      {/* Results Section */}
      {results && (
        <div className="rounded-lg border bg-card p-6 space-y-4">
          <h2 className="text-xl font-semibold">Analysis Results</h2>

          {results.message ? (
            <div className="rounded-md bg-yellow-50 border border-yellow-200 p-4">
              <p className="text-sm text-yellow-800">{results.message}</p>
              {results.ppe_items && Object.keys(results.ppe_items).length > 0 && (
                <div className="mt-3">
                  <p className="text-sm font-medium mb-2">PPE Detection (no face detected):</p>
                  <PPEStatus
                    ppeCompliant={results.ppe_compliant}
                    ppeItems={results.ppe_items}
                    ppeItemsAccuracy={results.ppe_items_accuracy}
                    showDetails={true}
                  />
                </div>
              )}
            </div>
          ) : (
            <div className="space-y-4">
              {results.results && results.results.map((result, index) => (
                <div
                  key={index}
                  className="rounded-lg border p-4 space-y-3"
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-lg font-semibold">
                        {result.name === 'Unknown' ? 'Unknown Person' : result.name}
                      </h3>
                      <p className="text-sm text-muted-foreground">
                        Recognition Confidence: {(result.confidence * 100).toFixed(1)}%
                      </p>
                      {!result.is_real && (
                        <span className="inline-block mt-1 px-2 py-1 text-xs font-semibold bg-orange-100 text-orange-800 rounded">
                          Spoof Detected
                        </span>
                      )}
                    </div>
                    <PPEStatus
                      ppeCompliant={result.ppe_compliant}
                      ppeItems={result.ppe_items}
                      ppeItemsAccuracy={result.ppe_items_accuracy}
                      showDetails={true}
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="font-medium">Face Recognition:</p>
                      <p className="text-muted-foreground">
                        {result.name} ({(result.confidence * 100).toFixed(1)}%)
                      </p>
                    </div>
                    <div>
                      <p className="font-medium">PPE Compliance:</p>
                      <p className={result.ppe_compliant ? 'text-green-600' : 'text-red-600'}>
                        {result.ppe_compliant ? 'Compliant' : 'Non-Compliant'}
                      </p>
                    </div>
                    <div>
                      <p className="font-medium">Helmet:</p>
                      <p className={result.helmet_detected ? 'text-green-600' : 'text-red-600'}>
                        {result.helmet_detected ? 'Detected' : 'Missing'}
                      </p>
                    </div>
                    <div>
                      <p className="font-medium">PPE Confidence:</p>
                      <p className="text-muted-foreground">
                        {(result.ppe_confidence * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>

                  {Object.keys(result.ppe_items || {}).length > 0 && (
                    <div className="mt-3 pt-3 border-t">
                      <p className="text-sm font-medium mb-2">PPE Items Detection Accuracy:</p>
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        {Object.entries(result.ppe_items).map(([item, detected]) => {
                          const accuracy = result.ppe_items_accuracy && result.ppe_items_accuracy[item] !== undefined
                            ? (result.ppe_items_accuracy[item] * 100).toFixed(1)
                            : 'N/A';
                          return (
                            <div
                              key={item}
                              className={`px-3 py-2 rounded border ${
                                detected
                                  ? 'bg-green-50 border-green-200 text-green-800'
                                  : 'bg-red-50 border-red-200 text-red-800'
                              }`}
                            >
                              <div className="flex items-center justify-between">
                                <span className="font-medium capitalize">{item}:</span>
                                <span className="font-semibold">{detected ? '✓' : '✗'}</span>
                              </div>
                              <div className="text-xs mt-1 opacity-75">
                                Accuracy: {accuracy}%
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              ))}

              {results.ppe_items && Object.keys(results.ppe_items).length > 0 && (
                <div className="mt-4 pt-4 border-t">
                  <p className="text-sm font-medium mb-2">Overall PPE Status:</p>
                  <PPEStatus
                    ppeCompliant={results.ppe_compliant}
                    ppeItems={results.ppe_items}
                    ppeItemsAccuracy={results.ppe_items_accuracy}
                    showDetails={true}
                  />
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ImageAnalysis;

