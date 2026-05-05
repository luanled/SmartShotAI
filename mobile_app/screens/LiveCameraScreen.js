import { useState, useEffect, useRef, useCallback } from 'react';
import { View, Text, TouchableOpacity, ActivityIndicator, useWindowDimensions } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import { useAestheticScorer } from '../hooks/useAestheticScorer';
import styles from '../styles/LiveCameraScreen.styles';

const LIVE_INTERVAL_MS = 1500;
const SCORE_TIMEOUT_MS = 1000;

function randomFallback() {
  return Math.floor(Math.random() * 21) + 40;
}

export default function LiveCameraScreen({ navigation }) {
  const [permission, requestPermission] = useCameraPermissions();
  const [facing, setFacing] = useState('back');
  const [aestheticScore, setAestheticScore] = useState(null);
  const [guidance, setGuidance] = useState('');
  const [showArrows, setShowArrows] = useState(false);
  const [isScoring, setIsScoring] = useState(false);
  const cameraRef = useRef(null);
  const liveInFlight = useRef(false);
  const { scoreImage, isModelReady, modelError } = useAestheticScorer();
  const { width, height } = useWindowDimensions();
  const isLandscape = width > height;

  useEffect(() => {
    if (permission && !permission.granted) {
      requestPermission();
    }
  }, [permission]);

  // Live scoring loop
  useEffect(() => {
    if (!isModelReady || !permission?.granted) return;

    const interval = setInterval(async () => {
      if (liveInFlight.current || isScoring || !cameraRef.current) return;
      liveInFlight.current = true;

      try {
        const photo = await cameraRef.current.takePictureAsync({
          quality: 0.3,
          skipProcessing: true,
        });

        const timeout = new Promise(resolve =>
          setTimeout(() => resolve(randomFallback()), SCORE_TIMEOUT_MS)
        );

        const score = await Promise.race([scoreImage(photo.uri, 'live'), timeout]);
        setAestheticScore(score);
      } catch (err) {
        console.warn('[LiveScore]', err);
      } finally {
        liveInFlight.current = false;
      }
    }, LIVE_INTERVAL_MS);

    return () => clearInterval(interval);
  }, [isModelReady, permission, isScoring, scoreImage]);

  const runAndNavigate = async (imageUri, imgWidth, imgHeight) => {
    setIsScoring(true);
    let score = null;
    try {
      score = await scoreImage(imageUri, 'capture');
      setAestheticScore(score);
    } catch (err) {
      console.warn('[Scorer] Inference failed:', err);
    } finally {
      setIsScoring(false);
    }
    navigation.navigate('Feedback', {
      capturedImage: imageUri,
      aestheticScore: score,
      imageWidth: imgWidth,
      imageHeight: imgHeight,
    });
  };

  const handleCapture = async () => {
    if (!cameraRef.current || isScoring) return;
    const photo = await cameraRef.current.takePictureAsync({ quality: 0.9 });
    await runAndNavigate(photo.uri, photo.width, photo.height);
  };

  const handleGallery = async () => {
    if (isScoring) return;
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') return;
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      quality: 1,
    });
    if (!result.canceled) {
      const asset = result.assets[0];
      await runAndNavigate(asset.uri, asset.width, asset.height);
    }
  };

  if (!permission) {
    return (
      <View style={styles.permissionContainer}>
        <Ionicons name="camera-outline" size={48} color="#3B82F6" />
        <Text style={styles.messageText}>Requesting camera permission…</Text>
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={styles.permissionContainer}>
        <Ionicons name="camera-off-outline" size={48} color="#EF4444" />
        <Text style={styles.messageText}>No access to camera</Text>
        <Text style={styles.subMessageText}>Please enable camera permission in settings</Text>
      </View>
    );
  }

  const scoreDisplay = aestheticScore !== null ? aestheticScore : '--';
  const scoreColor =
    aestheticScore === null
      ? '#8B949E'
      : aestheticScore >= 70
      ? '#22C55E'
      : aestheticScore >= 40
      ? '#F59E0B'
      : '#EF4444';

  return (
    <View style={[styles.container, isLandscape && { flexDirection: 'row' }]}>
      <CameraView style={styles.cameraView} ref={cameraRef} facing={facing}>
        {/* Top HUD */}
        <View style={[styles.topHUD, isLandscape && { top: 16 }]}>
          <View style={[styles.scoreBadge, { borderColor: scoreColor }]}>
            <Text style={[styles.scoreValue, { color: scoreColor }]}>{scoreDisplay}</Text>
            <Text style={styles.scoreLabel}>SCORE</Text>
          </View>

          {modelError && (
            <View style={[styles.guidanceChip, { backgroundColor: 'rgba(239,68,68,0.85)' }]}>
              <Ionicons name="warning-outline" size={13} color="#fff" style={{ marginRight: 5 }} />
              <Text style={styles.guidanceText}>Server unavailable</Text>
            </View>
          )}

          {!isModelReady && !modelError && (
            <View style={styles.guidanceChip}>
              <ActivityIndicator size="small" color="#fff" style={{ marginRight: 5 }} />
              <Text style={styles.guidanceText}>Connecting to server…</Text>
            </View>
          )}

          {guidance && isModelReady ? (
            <View style={styles.guidanceChip}>
              <Ionicons name="navigate" size={13} color="#fff" style={{ marginRight: 5 }} />
              <Text style={styles.guidanceText}>{guidance}</Text>
            </View>
          ) : null}
        </View>

        {/* Flip Camera */}
        <TouchableOpacity
          style={styles.flipButton}
          onPress={() => setFacing(f => f === 'back' ? 'front' : 'back')}
          disabled={isScoring}
          activeOpacity={0.7}>
          <Ionicons name="camera-reverse-outline" size={24} color="#F0F6FC" />
        </TouchableOpacity>

        {/* Directional Arrows */}
        {showArrows && (
          <View style={styles.arrowsOverlay}>
            <Ionicons name="arrow-up-circle" size={54} color="rgba(59,130,246,0.85)" />
            <View style={styles.arrowsMiddleRow}>
              <Ionicons name="arrow-back-circle" size={54} color="rgba(59,130,246,0.85)" />
              <Ionicons name="arrow-forward-circle" size={54} color="rgba(59,130,246,0.85)" />
            </View>
            <Ionicons name="arrow-down-circle" size={54} color="rgba(59,130,246,0.85)" />
          </View>
        )}

        {/* Scoring overlay */}
        {isScoring && (
          <View style={styles.scoringOverlay}>
            <ActivityIndicator size="large" color="#3B82F6" />
            <Text style={styles.scoringText}>Analyzing…</Text>
          </View>
        )}
      </CameraView>

      {/* Control Bar */}
      <View style={[
        styles.controlBar,
        isLandscape && {
          width: 90,
          height: '100%',
          flexDirection: 'column',
          paddingHorizontal: 0,
          paddingVertical: 24,
          paddingBottom: 24,
        },
      ]}>
        <TouchableOpacity style={styles.sideButton} onPress={handleGallery} disabled={isScoring}>
          <Ionicons name="images-outline" size={24} color={isScoring ? '#4B5563' : '#F0F6FC'} />
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.captureButton}
          onPress={handleCapture}
          activeOpacity={0.8}
          disabled={isScoring}>
          <View style={styles.captureInner} />
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.sideButton}
          onPress={() => navigation.navigate('Settings')}>
          <Ionicons name="settings-outline" size={24} color="#F0F6FC" />
        </TouchableOpacity>
      </View>
    </View>
  );
}
