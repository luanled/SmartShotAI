import { useState, useEffect, useCallback } from 'react';
import { BACKEND_URL } from '../config';
import { useModel } from '../context/ModelContext';

export function useAestheticScorer() {
  const [isModelReady, setIsModelReady] = useState(false);
  const [modelError, setModelError] = useState(false);
  const { selectedModel } = useModel();

  useEffect(() => {
    setIsModelReady(false);
    setModelError(false);
    fetch(`${BACKEND_URL}/health`)
      .then(r => {
        if (r.ok) setIsModelReady(true);
        else setModelError(true);
      })
      .catch(() => setModelError(true));
  }, []);

  // source: 'live' | 'capture'
  const scoreImage = useCallback(async (imageUri, source = 'live') => {
    const formData = new FormData();
    formData.append('file', { uri: imageUri, name: 'photo.jpg', type: 'image/jpeg' });
    formData.append('model', selectedModel);
    formData.append('source', source);

    const response = await fetch(`${BACKEND_URL}/score`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) throw new Error(`Server error: ${response.status}`);
    const { score } = await response.json();
    return score;
  }, [selectedModel]);

  return { scoreImage, isModelReady, modelError };
}
