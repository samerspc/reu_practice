import React, { useState } from 'react';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setPrediction(null);
    setError(null);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select a file first!');
      return;
    }

    setLoading(true);
    setPrediction(null);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://localhost:5001/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setPrediction(data);
    } catch (e) {
      setError(`Failed to fetch prediction: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Food Recognition App</h1>
        <input type="file" onChange={handleFileChange} />
        <button onClick={handleUpload} disabled={!selectedFile || loading}>
          {loading ? 'Predicting...' : 'Upload and Predict'}
        </button>

        {error && <p className="error-message">{error}</p>}

        {prediction && (
          <div className="prediction-result">
            <h2>Prediction: {prediction.predicted_food}</h2>
            <p>Confidence: {(prediction.confidence * 100).toFixed(2)}%</p>
            {selectedFile && (
              <img
                src={URL.createObjectURL(selectedFile)}
                alt="Uploaded Food"
                className="uploaded-image"
              />
            )}
          </div>
        )}
      </header>
    </div>
  );
}

export default App;
