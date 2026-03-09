import { useState } from 'react';
import { View, Text, Image, TouchableOpacity, Alert } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as MediaLibrary from 'expo-media-library';
import styles from '../styles/FeedbackScreen.styles';

export default function FeedbackScreen({ route, navigation }) {
  const { aestheticScore, capturedImage } = route.params || {};
  const [feedback, setFeedback] = useState(null);

  const score = aestheticScore || 80;
  const scoreColor = score >= 70 ? '#22C55E' : score >= 40 ? '#F59E0B' : '#EF4444';

  const handleSave = async () => {
    const { status } = await MediaLibrary.requestPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert(
        'Permission Required',
        'Please allow access to your photo library in Settings.',
      );
      return;
    }
    try {
      await MediaLibrary.saveToLibraryAsync(capturedImage);
      Alert.alert('Photo Saved', 'Your photo has been saved to the gallery.', [
        { text: 'OK', onPress: () => navigation.navigate('LiveCamera') },
      ]);
    } catch {
      Alert.alert('Error', 'Failed to save the photo. Please try again.');
    }
  };

  const handleDiscard = () => {
    Alert.alert('Discard Photo', 'Are you sure you want to discard this photo?', [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Discard', style: 'destructive', onPress: () => navigation.navigate('LiveCamera') },
    ]);
  };

  return (
    <View style={styles.container}>
      {/* Photo Preview */}
      <View style={styles.photoArea}>
        <View style={styles.photoPlaceholder}>
          {capturedImage ? (
            <Image source={{ uri: capturedImage }} style={styles.photo} resizeMode="cover" />
          ) : (
            <Ionicons name="image-outline" size={64} color="#30363D" />
          )}
          <View style={[styles.scorePill, { borderColor: scoreColor }]}>
            <Text style={[styles.scorePillText, { color: scoreColor }]}>Score: {score}%</Text>
          </View>
        </View>

      </View>

      {/* Bottom Sheet */}
      <View style={styles.sheet}>
        <Text style={styles.sheetTitle}>How was the guidance?</Text>

        <View style={styles.thumbsRow}>
          <TouchableOpacity
            style={[styles.thumbBtn, feedback === 'positive' && styles.thumbBtnPositive]}
            onPress={() => setFeedback('positive')}
            activeOpacity={0.8}>
            <Ionicons
              name="thumbs-up"
              size={32}
              color={feedback === 'positive' ? '#2563EB' : '#8B949E'}
            />
            <Text style={[styles.thumbLabel, feedback === 'positive' && styles.thumbLabelPositive]}>
              Helpful
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.thumbBtn, feedback === 'negative' && styles.thumbBtnNegative]}
            onPress={() => setFeedback('negative')}
            activeOpacity={0.8}>
            <Ionicons
              name="thumbs-down"
              size={32}
              color={feedback === 'negative' ? '#EF4444' : '#8B949E'}
            />
            <Text style={[styles.thumbLabel, feedback === 'negative' && styles.thumbLabelNegative]}>
              Not Helpful
            </Text>
          </TouchableOpacity>
        </View>

        <View style={styles.actionRow}>
          <TouchableOpacity style={styles.discardBtn} onPress={handleDiscard} activeOpacity={0.8}>
            <Text style={styles.discardBtnText}>Discard</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.saveBtn} onPress={handleSave} activeOpacity={0.8}>
            <Ionicons name="download-outline" size={18} color="#fff" style={{ marginRight: 6 }} />
            <Text style={styles.saveBtnText}>Save Photo</Text>
          </TouchableOpacity>
        </View>
      </View>
    </View>
  );
}
