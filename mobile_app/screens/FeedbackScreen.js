import { View, Text, Image, TouchableOpacity, Alert, useWindowDimensions } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as MediaLibrary from 'expo-media-library';
import styles from '../styles/FeedbackScreen.styles';

export default function FeedbackScreen({ route, navigation }) {
  const { aestheticScore, capturedImage, imageWidth: imgW, imageHeight: imgH } = route.params || {};
  const { width, height } = useWindowDimensions();
  const isLandscapeScreen = width > height;
  const isLandscapeImage = imgW && imgH && imgW > imgH;

  const score = aestheticScore;
  const hasScore = score !== null && score !== undefined;
  const scoreColor = !hasScore
    ? '#8B949E'
    : score >= 70
    ? '#22C55E'
    : score >= 40
    ? '#F59E0B'
    : '#EF4444';

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
    <View style={[styles.container, isLandscapeScreen && { flexDirection: 'row' }]}>
      {/* Photo Preview */}
      <View style={[styles.photoArea, isLandscapeScreen && { flex: 3 }]}>
        <View style={styles.photoPlaceholder}>
          {capturedImage ? (
            <Image
              source={{ uri: capturedImage }}
              style={styles.photo}
              resizeMode={isLandscapeImage ? 'contain' : 'cover'}
            />
          ) : (
            <Ionicons name="image-outline" size={64} color="#30363D" />
          )}
          <View style={[styles.scorePill, { borderColor: scoreColor }]}>
            <Text style={[styles.scorePillText, { color: scoreColor }]}>
              {hasScore ? `Score: ${score}%` : 'Score: N/A'}
            </Text>
          </View>
        </View>
      </View>

      {/* Sheet — bottom in portrait, right panel in landscape */}
      <View style={[
        styles.sheet,
        isLandscapeScreen && {
          flex: 2,
          borderTopLeftRadius: 24,
          borderTopRightRadius: 0,
          borderBottomLeftRadius: 0,
          borderTopWidth: 0,
          borderLeftWidth: 1,
          borderLeftColor: '#21262D',
          justifyContent: 'center',
          paddingTop: 24,
          paddingBottom: 24,
        },
      ]}>
        <View style={[styles.actionRow, isLandscapeScreen && { flexDirection: 'column' }]}>
          <TouchableOpacity
            style={[styles.discardBtn, isLandscapeScreen && { flex: undefined }]}
            onPress={handleDiscard}
            activeOpacity={0.8}>
            <Text style={styles.discardBtnText}>Discard</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.saveBtn, isLandscapeScreen && { flex: undefined }]}
            onPress={handleSave}
            activeOpacity={0.8}>
            <Ionicons name="download-outline" size={18} color="#fff" style={{ marginRight: 6 }} />
            <Text style={styles.saveBtnText}>Save Photo</Text>
          </TouchableOpacity>
        </View>
      </View>
    </View>
  );
}
