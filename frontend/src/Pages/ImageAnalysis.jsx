import { useState } from 'react';
import flaskAxios from '../utils/flaskAxios';
import PPEStatus from '../Components/PPEStatus';

const ImageAnalysis = () => {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Array to store all analyzed images
  const [analyzedImages, setAnalyzedImages] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);

  const FLASK_URL = import.meta.env.VITE_FLASK_URL || 'http://localhost:5000';

  const handleFileChange = (e) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) {
      // Store all selected files
      setSelectedFiles(files);
      setError(null);
      
      // Create preview for the first file
      const firstFile = files[0];
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
      };
      reader.readAsDataURL(firstFile);
    }
  };

  const handleAnalyze = async () => {
    if (selectedFiles.length === 0) {
      setError('Please select at least one image first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const newAnalyzedImages = [];
      
      // Process each selected file
      for (let i = 0; i < selectedFiles.length; i++) {
        const file = selectedFiles[i];
        
        // Create preview for this file
        const filePreview = await new Promise((resolve) => {
          const reader = new FileReader();
          reader.onloadend = () => resolve(reader.result);
          reader.readAsDataURL(file);
        });

        // Analyze the image
        const formData = new FormData();
        formData.append('image', file);

        const response = await flaskAxios.post(`${FLASK_URL}/api/analyze-image`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });

        // Create analysis result object
        const analysisResult = {
          id: Date.now() + i, // Unique ID for this analysis
          fileName: file.name,
          originalImage: filePreview,
          results: response.data,
          analyzedAt: new Date().toISOString()
        };

        newAnalyzedImages.push(analysisResult);
      }

      // Add all analyzed images to the array
      setAnalyzedImages(prev => [...prev, ...newAnalyzedImages]);
      
      // Set current index to the first newly added image
      setCurrentIndex(analyzedImages.length);
      
      // Clear selected files and preview
      setSelectedFiles([]);
      setPreview(null);
      
      // Reset file input
      const fileInput = document.querySelector('input[type="file"]');
      if (fileInput) {
        fileInput.value = '';
      }
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
    setSelectedFiles([]);
    setPreview(null);
    setError(null);
    // Reset file input
    const fileInput = document.querySelector('input[type="file"]');
    if (fileInput) {
      fileInput.value = '';
    }
  };

  const handleClearAll = () => {
    setAnalyzedImages([]);
    setCurrentIndex(0);
    setSelectedFiles([]);
    setPreview(null);
    setError(null);
    // Reset file input
    const fileInput = document.querySelector('input[type="file"]');
    if (fileInput) {
      fileInput.value = '';
    }
  };

  const handlePrevious = () => {
    if (currentIndex > 0) {
      setCurrentIndex(currentIndex - 1);
    }
  };

  const handleNext = () => {
    if (currentIndex < analyzedImages.length - 1) {
      setCurrentIndex(currentIndex + 1);
    }
  };

  // Get current analysis result
  const currentAnalysis = analyzedImages.length > 0 ? analyzedImages[currentIndex] : null;
  const results = currentAnalysis ? currentAnalysis.results : null;
  const currentPreview = currentAnalysis ? currentAnalysis.originalImage : preview;

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex flex-col space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">Image Analysis</h1>
        <p className="text-muted-foreground">
          Upload and analyze multiple images to identify employees and detect PPE compliance. Use the navigation buttons to browse through analyzed images.
        </p>
      </div>

      {/* Upload Controls */}
      <div className="rounded-lg border bg-card p-4">
        <div className="flex items-center gap-4">
          <div className="flex-1">
            <label className="block text-sm font-medium mb-2">
              Select Image
            </label>
            <input
              type="file"
              accept="image/*"
              multiple
              onChange={handleFileChange}
              className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
            />
            {selectedFiles.length > 0 && (
              <p className="text-xs text-muted-foreground mt-2">
                {selectedFiles.length} image(s) selected: {selectedFiles.map(f => f.name).join(', ')}
              </p>
            )}
          </div>
          <div className="flex gap-4 items-end">
            <button
              onClick={handleAnalyze}
              disabled={selectedFiles.length === 0 || loading}
              className={`px-6 py-2 rounded-md font-medium ${
                selectedFiles.length === 0 || loading
                  ? 'bg-gray-400 text-white cursor-not-allowed'
                  : 'bg-blue-500 text-white hover:bg-blue-600'
              }`}
            >
              {loading ? `Analyzing ${selectedFiles.length} image(s)...` : `Analyze ${selectedFiles.length > 0 ? `${selectedFiles.length} ` : ''}Image(s)`}
            </button>

            {selectedFiles.length > 0 && (
              <button
                onClick={handleClear}
                className="px-6 py-2 rounded-md font-medium bg-gray-500 text-white hover:bg-gray-600"
              >
                Clear Selection
              </button>
            )}
            {analyzedImages.length > 0 && (
              <button
                onClick={handleClearAll}
                className="px-6 py-2 rounded-md font-medium bg-red-500 text-white hover:bg-red-600"
              >
                Clear All
              </button>
            )}
          </div>
        </div>

        {error && (
          <div className="mt-4 rounded-md bg-red-50 border border-red-200 p-3">
            <p className="text-sm text-red-800">{error}</p>
          </div>
        )}
      </div>

      {/* Main Content: Image on Left, Results on Right */}
      {(currentPreview || results || analyzedImages.length > 0) && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Side - Image Display */}
          <div className="rounded-lg border bg-card p-4 flex flex-col">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold">
                {results && results.annotated_image ? 'Annotated Image' : 'Image Preview'}
              </h2>
              {analyzedImages.length > 1 && (
                <div className="flex items-center gap-2">
                  <span className="text-sm text-muted-foreground">
                    {currentIndex + 1} / {analyzedImages.length}
                  </span>
                </div>
              )}
            </div>
            
            {/* Carousel Navigation */}
            {analyzedImages.length > 1 && (
              <div className="flex items-center justify-between mb-4">
                <button
                  onClick={handlePrevious}
                  disabled={currentIndex === 0}
                  className={`px-4 py-2 rounded-md font-medium ${
                    currentIndex === 0
                      ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                      : 'bg-blue-500 text-white hover:bg-blue-600'
                  }`}
                >
                  ← Previous
                </button>
                <span className="text-sm text-muted-foreground font-medium">
                  {currentAnalysis?.fileName || 'Image'}
                </span>
                <button
                  onClick={handleNext}
                  disabled={currentIndex === analyzedImages.length - 1}
                  className={`px-4 py-2 rounded-md font-medium ${
                    currentIndex === analyzedImages.length - 1
                      ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                      : 'bg-blue-500 text-white hover:bg-blue-600'
                  }`}
                >
                  Next →
                </button>
              </div>
            )}
            
            <div className="flex-1 overflow-y-auto pr-2" style={{ maxHeight: 'calc(100vh - 300px)' }}>
              {results && results.annotated_image ? (
                // Show annotated image when analysis is done
                <div>
                  <p className="text-sm font-medium mb-2 text-muted-foreground">
                    Detection results with bounding boxes and person identification
                  </p>
                  <div className="bg-black rounded-lg overflow-hidden flex items-center justify-center" style={{ minHeight: '200px' }}>
                    <img
                      src={results.annotated_image}
                      alt="Annotated Analysis"
                      className="max-w-full max-h-full object-contain rounded-lg"
                    />
                  </div>
                </div>
              ) : currentPreview ? (
                // Show original image only when no analysis results yet
                <div>
                  <p className="text-sm font-medium mb-2 text-muted-foreground">Original Image:</p>
                  <div className="bg-black rounded-lg overflow-hidden flex items-center justify-center" style={{ minHeight: '200px' }}>
                    <img
                      src={currentPreview}
                      alt="Preview"
                      className="max-w-full max-h-full object-contain rounded-lg"
                    />
                  </div>
                </div>
              ) : (
                <div className="flex items-center justify-center h-full text-muted-foreground">
                  <p className="text-sm">No image selected</p>
                </div>
              )}
            </div>
          </div>

          {/* Right Side - Analysis Results */}
          <div className="rounded-lg border bg-card p-4 flex flex-col">
            <h2 className="text-xl font-semibold mb-4">Analysis Results</h2>
            <div className="flex-1 overflow-y-auto space-y-4 pr-2" style={{ maxHeight: 'calc(100vh - 300px)' }}>
              {results ? (
                results.message ? (
                  // No face detected - use same layout as face detected
                  <div className="rounded-lg border p-4 space-y-4">
                    {/* Face Recognition, PPE Compliance, PPE Confidence */}
                    <div className="grid grid-cols-3 gap-4 text-sm">
                      <div>
                        <p className="font-medium">Face Recognition:</p>
                        <p className="text-muted-foreground">
                          No face detected
                        </p>
                      </div>
                      <div>
                        <p className="font-medium">PPE Compliance:</p>
                        <p className={results.ppe_compliant ? 'text-green-600' : 'text-red-600'}>
                          {results.ppe_compliant ? 'Compliant' : 'Non-Compliant'}
                        </p>
                      </div>
                      <div>
                        <p className="font-medium">PPE Confidence:</p>
                        <p className="text-muted-foreground">
                          {results.ppe_items_accuracy && Object.keys(results.ppe_items_accuracy).length > 0
                            ? (Object.values(results.ppe_items_accuracy).reduce((sum, acc) => sum + acc, 0) / Object.keys(results.ppe_items_accuracy).length * 100).toFixed(1)
                            : '0.0'}%
                        </p>
                      </div>
                    </div>

                    {/* PPE Items Detection Accuracy - Square Boxes */}
                    {results.ppe_items && Object.keys(results.ppe_items).length > 0 && (
                      <div className="mt-4 pt-4 border-t">
                        <p className="text-sm font-medium mb-3">PPE Items Detection:</p>
                        <div className="grid grid-cols-2 gap-3">
                          {Object.entries(results.ppe_items).map(([item, detected]) => {
                            const accuracy = results.ppe_items_accuracy && results.ppe_items_accuracy[item] !== undefined
                              ? (results.ppe_items_accuracy[item] * 100).toFixed(1)
                              : '0.0';
                            return (
                              <div
                                key={item}
                                className={`px-4 py-3 rounded-lg border-2 ${
                                  detected
                                    ? 'bg-green-50 border-green-300 text-green-800'
                                    : 'bg-red-50 border-red-300 text-red-800'
                                }`}
                              >
                                <div className="flex items-center justify-between mb-1">
                                  <span className="font-semibold capitalize text-base">{item}</span>
                                  <span className="text-xl font-bold">{detected ? '✓' : '✗'}</span>
                                </div>
                                <div className="text-xs opacity-75">
                                  Accuracy: {accuracy}%
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="space-y-4">
                    {results.results && results.results.map((result, index) => (
                      <div
                        key={index}
                        className="rounded-lg border p-4 space-y-4"
                      >
                        {/* Person Recognition Header */}
                        {result.name && result.name !== 'Unknown' && (
                          <div className="text-center py-3 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border border-blue-200">
                            <p className="text-xs text-blue-600 font-medium mb-1">Now you are seeing</p>
                            <p className="text-2xl font-bold text-blue-700">
                              {result.name}
                            </p>
                          </div>
                        )}

                        {/* Face Recognition, PPE Compliance, PPE Confidence */}
                        <div className="grid grid-cols-3 gap-4 text-sm">
                          <div>
                            <p className="font-medium">Face Recognition:</p>
                            <p className="text-muted-foreground">
                              {(result.confidence * 100).toFixed(1)}%
                            </p>
                          </div>
                          <div>
                            <p className="font-medium">PPE Compliance:</p>
                            <p className={result.ppe_compliant ? 'text-green-600' : 'text-red-600'}>
                              {result.ppe_compliant ? 'Compliant' : 'Non-Compliant'}
                            </p>
                          </div>
                          <div>
                            <p className="font-medium">PPE Confidence:</p>
                            <p className="text-muted-foreground">
                              {(result.ppe_confidence * 100).toFixed(1)}%
                            </p>
                          </div>
                        </div>

                        {/* PPE Items Detection Accuracy - Square Boxes */}
                        {Object.keys(result.ppe_items || {}).length > 0 && (
                          <div className="mt-4 pt-4 border-t">
                            <p className="text-sm font-medium mb-3">PPE Items Detection:</p>
                            <div className="grid grid-cols-2 gap-3">
                              {Object.entries(result.ppe_items).map(([item, detected]) => {
                                const accuracy = result.ppe_items_accuracy && result.ppe_items_accuracy[item] !== undefined
                                  ? (result.ppe_items_accuracy[item] * 100).toFixed(1)
                                  : '0.0';
                                return (
                                  <div
                                    key={item}
                                    className={`px-4 py-3 rounded-lg border-2 ${
                                      detected
                                        ? 'bg-green-50 border-green-300 text-green-800'
                                        : 'bg-red-50 border-red-300 text-red-800'
                                    }`}
                                  >
                                    <div className="flex items-center justify-between mb-1">
                                      <span className="font-semibold capitalize text-base">{item}</span>
                                      <span className="text-xl font-bold">{detected ? '✓' : '✗'}</span>
                                    </div>
                                    <div className="text-xs opacity-75">
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
                  </div>
                )
              ) : (
                <p className="text-sm text-muted-foreground text-center py-8">
                  Upload and analyze an image to see results here
                </p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageAnalysis;

