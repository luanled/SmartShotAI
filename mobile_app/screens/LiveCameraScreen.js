import { useState, useEffect, useRef } from 'react';
import { View, Text, TouchableOpacity } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import styles from '../styles/LiveCameraScreen.styles';

export default function LiveCameraScreen({ navigation }) {
  const [permission, requestPermission] = useCameraPermissions();
  const [aestheticScore, setAestheticScore] = useState(0);
  const [guidance, setGuidance] = useState('');
  const [showArrows, setShowArrows] = useState(false);
  const cameraRef = useRef(null);

  useEffect(() => {
    if (permission && !permission.granted) {
      requestPermission();
    }
  }, [permission]);

  const handleCapture = async () => {
    if (cameraRef.current) {
      const photo = await cameraRef.current.takePictureAsync();
      navigation.navigate('Feedback', {
        capturedImage: photo.uri,
        aestheticScore,
      });
    }
  };

  const handleGallery = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') return;
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      quality: 1,
    });
    if (!result.canceled) {
      navigation.navigate('Feedback', {
        capturedImage: result.assets[0].uri,
        aestheticScore,
      });
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

  const scoreColor =
    aestheticScore >= 70 ? '#22C55E' : aestheticScore >= 40 ? '#F59E0B' : '#EF4444';

  return (
    <View style={styles.container}>
      <CameraView style={styles.cameraView} ref={cameraRef}>
        {/* Top HUD */}
        <View style={styles.topHUD}>
          <View style={[styles.scoreBadge, { borderColor: scoreColor }]}>
            <Text style={[styles.scoreValue, { color: scoreColor }]}>{aestheticScore}</Text>
            <Text style={styles.scoreLabel}>SCORE</Text>
          </View>

          {guidance ? (
            <View style={styles.guidanceChip}>
              <Ionicons name="navigate" size={13} color="#fff" style={{ marginRight: 5 }} />
              <Text style={styles.guidanceText}>{guidance}</Text>
            </View>
          ) : null}
        </View>

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
      </CameraView>

      {/* Control Bar */}
      <View style={styles.controlBar}>
        <TouchableOpacity style={styles.sideButton} onPress={handleGallery}>
          <Ionicons name="images-outline" size={24} color="#F0F6FC" />
        </TouchableOpacity>

        <TouchableOpacity style={styles.captureButton} onPress={handleCapture} activeOpacity={0.8}>
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
