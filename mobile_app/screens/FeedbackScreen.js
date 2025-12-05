import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Alert } from 'react-native';

export default function FeedbackScreen({ route, navigation }) {
  const { aestheticScore } = route.params || {};
  const [feedback, setFeedback] = useState(null);

  const handleThumbsUp = () => {
    setFeedback('positive');
  };

  const handleThumbsDown = () => {
    setFeedback('negative');
  };

  const handleSave = () => {
    Alert.alert('Photo Saved', 'Your photo has been saved to the gallery.', [
      { text: 'OK', onPress: () => navigation.navigate('LiveCamera') },
    ]);
  };

  const handleDiscard = () => {
    Alert.alert('Discard Photo', 'Are you sure you want to discard this photo?', [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Discard', style: 'destructive', onPress: () => navigation.navigate('LiveCamera') },
    ]);
  };

  return (
    <View style={styles.container}>
      <View style={styles.photoContainer}>
        <View style={styles.photoPlaceholder}>
          <Text style={styles.photoIcon}>🖼️</Text>
          <Text style={styles.photoText}>Captured Photo</Text>
          <Text style={styles.photoScore}>Score: {aestheticScore || 80}%</Text>
        </View>
      </View>

      <View style={styles.feedbackSection}>
        <Text style={styles.feedbackTitle}>How was the guidance?</Text>
        <View style={styles.thumbsContainer}>
          <TouchableOpacity
            style={[styles.thumbButton, feedback === 'positive' && styles.thumbButtonActive]}
            onPress={handleThumbsUp}>
            <Text style={styles.thumbIcon}>👍</Text>
            <Text style={styles.thumbLabel}>Good</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[styles.thumbButton, feedback === 'negative' && styles.thumbButtonActive]}
            onPress={handleThumbsDown}>
            <Text style={styles.thumbIcon}>👎</Text>
            <Text style={styles.thumbLabel}>Not Helpful</Text>
          </TouchableOpacity>
        </View>
      </View>

      <View style={styles.actionSection}>
        <TouchableOpacity style={[styles.actionButton, styles.saveButton]} onPress={handleSave}>
          <Text style={styles.actionButtonText}>Save</Text>
        </TouchableOpacity>
        <TouchableOpacity style={[styles.actionButton, styles.discardButton]} onPress={handleDiscard}>
          <Text style={styles.actionButtonText}>Discard</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.bottomNav}>
        <TouchableOpacity style={styles.navButton} onPress={() => navigation.navigate('LiveCamera')}>
          <Text style={styles.navIcon}>🖼️</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.navButton} onPress={() => navigation.navigate('LiveCamera')}>
          <Text style={styles.navIcon}>📷</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.navButton} onPress={() => navigation.navigate('Settings')}>
          <Text style={styles.navIcon}>⚙️</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#D6E9F8',
  },
  photoContainer: {
    flex: 1,
    padding: 20,
    justifyContent: 'center',
    alignItems: 'center',
  },
  photoPlaceholder: {
    width: '100%',
    height: '90%',
    backgroundColor: '#8B7355',
    borderRadius: 10,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 3,
    borderColor: '#5D4E37',
  },
  photoIcon: {
    fontSize: 64,
    marginBottom: 10,
  },
  photoText: {
    fontSize: 18,
    color: '#FFFFFF',
    fontWeight: '600',
    marginBottom: 10,
  },
  photoScore: {
    fontSize: 16,
    color: '#E0E0E0',
  },
  feedbackSection: {
    paddingHorizontal: 20,
    paddingVertical: 15,
    backgroundColor: '#FFFFFF',
  },
  feedbackTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#424242',
    marginBottom: 15,
    textAlign: 'center',
  },
  thumbsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 10,
  },
  thumbButton: {
    alignItems: 'center',
    padding: 15,
    borderRadius: 10,
    backgroundColor: '#F5F5F5',
    minWidth: 120,
  },
  thumbButtonActive: {
    backgroundColor: '#E3F2FD',
    borderWidth: 2,
    borderColor: '#1E88E5',
  },
  thumbIcon: {
    fontSize: 40,
    marginBottom: 5,
  },
  thumbLabel: {
    fontSize: 14,
    color: '#616161',
    fontWeight: '500',
  },
  actionSection: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingHorizontal: 20,
    paddingVertical: 20,
    backgroundColor: '#FFFFFF',
  },
  actionButton: {
    flex: 1,
    paddingVertical: 15,
    borderRadius: 8,
    marginHorizontal: 10,
    alignItems: 'center',
  },
  saveButton: {
    backgroundColor: '#81C784',
  },
  discardButton: {
    backgroundColor: '#E57373',
  },
  actionButtonText: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#FFFFFF',
  },
  bottomNav: {
    height: 80,
    backgroundColor: '#D6E9F8',
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    borderTopWidth: 1,
    borderTopColor: '#B0BEC5',
  },
  navButton: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#FFFFFF',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: '#1E88E5',
  },
  navIcon: {
    fontSize: 28,
  },
});