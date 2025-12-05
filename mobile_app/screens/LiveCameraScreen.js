import React, { useState, useEffect, useRef } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Alert } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';

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
        aestheticScore: aestheticScore 
      });
    }
  };

  const handleSettings = () => {
    navigation.navigate('Settings');
  };

  if (!permission) {
    return (
      <View style={styles.container}>
        <Text style={styles.messageText}>Requesting camera permission...</Text>
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.messageText}>No access to camera</Text>
        <Text style={styles.subMessageText}>Please enable camera permission in settings</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <CameraView style={styles.cameraView} ref={cameraRef}>
        {/* Aesthetic Score Badge with Ring */}
        <View style={styles.scoreBadgeContainer}>
          <View style={styles.scoreRing}>
            <View style={styles.scoreBadge}>
              <Text style={styles.scorePercentage}>{aestheticScore}</Text>
            </View>
          </View>
        </View>

        {/* Guidance Message */}
        {guidance && (
          <View style={styles.guidanceContainer}>
            <Text style={styles.guidanceText}>{guidance}</Text>
          </View>
        )}

        {/* Directional Arrows - only show when model provides policy feedback */}
        {showArrows && (
          <View style={styles.arrowsContainer}>
            <TouchableOpacity style={styles.arrowUp}>
              <Text style={styles.arrowIcon}>⬆️</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.arrowLeft}>
              <Text style={styles.arrowIcon}>⬅️</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.arrowRight}>
              <Text style={styles.arrowIcon}>➡️</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.arrowDown}>
              <Text style={styles.arrowIcon}>⬇️</Text>
            </TouchableOpacity>
          </View>
        )}
      </CameraView>

      <View style={styles.controlBar}>
        <TouchableOpacity style={styles.controlButton}>
          <Text style={styles.controlIcon}>🖼️</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.captureButton} onPress={handleCapture}>
          <View style={styles.captureButtonInner}>
            <Text style={styles.captureIcon}>📷</Text>
          </View>
        </TouchableOpacity>
        <TouchableOpacity style={styles.controlButton} onPress={handleSettings}>
          <Text style={styles.controlIcon}>⚙️</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
    justifyContent: 'center',
    alignItems: 'center',
  },
  messageText: {
    fontSize: 18,
    color: '#FFF',
    textAlign: 'center',
    marginBottom: 10,
  },
  subMessageText: {
    fontSize: 14,
    color: '#BBB',
    textAlign: 'center',
  },
  cameraView: {
    flex: 1,
    width: '100%',
    position: 'relative',
  },
  scoreBadgeContainer: {
    position: 'absolute',
    top: 50,
    left: 20,
  },
  scoreRing: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 3,
    borderColor: 'rgba(30, 136, 229, 0.8)',
  },
  scoreBadge: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: 'rgba(30, 136, 229, 0.95)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  scorePercentage: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#FFFFFF',
  },
  guidanceContainer: {
    position: 'absolute',
    top: 50,
    right: 20,
    backgroundColor: 'rgba(76, 175, 80, 0.9)',
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 20,
  },
  guidanceText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#FFFFFF',
  },
  arrowsContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
  },
  arrowUp: {
    position: 'absolute',
    top: '25%',
    alignSelf: 'center',
  },
  arrowDown: {
    position: 'absolute',
    bottom: '30%',
    alignSelf: 'center',
  },
  arrowLeft: {
    position: 'absolute',
    left: '15%',
    top: '50%',
  },
  arrowRight: {
    position: 'absolute',
    right: '15%',
    top: '50%',
  },
  arrowIcon: {
    fontSize: 40,
    opacity: 0.8,
  },
  controlBar: {
    height: 120,
    backgroundColor: '#D6E9F8',
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    paddingHorizontal: 40,
  },
  controlButton: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#FFFFFF',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: '#1E88E5',
  },
  controlIcon: {
    fontSize: 28,
  },
  captureButton: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: '#00BCD4',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 4,
    borderColor: '#FFFFFF',
  },
  captureButtonInner: {
    width: 68,
    height: 68,
    borderRadius: 34,
    backgroundColor: '#00ACC1',
    justifyContent: 'center',
    alignItems: 'center',
  },
  captureIcon: {
    fontSize: 32,
  },
});