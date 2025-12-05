import React, { useState } from 'react';
import { View, Text, StyleSheet, Switch, ScrollView, TouchableOpacity } from 'react-native';

export default function SettingsScreen({ navigation }) {
  const [voiceGuidance, setVoiceGuidance] = useState(true);
  const [vibrationFeedback, setVibrationFeedback] = useState(true);
  const [personalizationLevel, setPersonalizationLevel] = useState(50);

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity style={styles.backButton} onPress={() => navigation.goBack()}>
          <Text style={styles.backIcon}>←</Text>
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Settings & Personalization</Text>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Feedback Settings</Text>
        
        <View style={styles.settingRow}>
          <View style={styles.settingInfo}>
            <Text style={styles.settingLabel}>Voice guidance</Text>
            <Text style={styles.settingDescription}>Hear audio directions like "move left"</Text>
          </View>
          <Switch
            value={voiceGuidance}
            onValueChange={setVoiceGuidance}
            trackColor={{ false: '#D0D0D0', true: '#64B5F6' }}
            thumbColor={voiceGuidance ? '#1E88E5' : '#F5F5F5'}
          />
        </View>

        <View style={styles.settingRow}>
          <View style={styles.settingInfo}>
            <Text style={styles.settingLabel}>Vibration feedback</Text>
            <Text style={styles.settingDescription}>Feel haptic feedback for guidance</Text>
          </View>
          <Switch
            value={vibrationFeedback}
            onValueChange={setVibrationFeedback}
            trackColor={{ false: '#D0D0D0', true: '#64B5F6' }}
            thumbColor={vibrationFeedback ? '#1E88E5' : '#F5F5F5'}
          />
        </View>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Personalization</Text>
        <View style={styles.sliderContainer}>
          <Text style={styles.settingLabel}>Personalization Level</Text>
          <Text style={styles.settingDescription}>How much the AI adapts to your style</Text>
          <Text style={styles.sliderValue}>{personalizationLevel}%</Text>
        </View>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Info</Text>
        <View style={styles.infoRow}>
          <Text style={styles.infoLabel}>Model Version:</Text>
          <Text style={styles.infoValue}>1.0.0</Text>
        </View>
        <View style={styles.infoRow}>
          <Text style={styles.infoLabel}>Build Version:</Text>
          <Text style={styles.infoValue}>01</Text>
        </View>
      </View>

      <View style={styles.section}>
        <TouchableOpacity style={styles.helpButton}>
          <Text style={styles.helpButtonText}>About / Help</Text>
          <Text style={styles.helpIcon}>→</Text>
        </TouchableOpacity>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5',
  },
  header: {
    backgroundColor: '#1E88E5',
    paddingTop: 50,
    paddingBottom: 20,
    paddingHorizontal: 20,
    flexDirection: 'row',
    alignItems: 'center',
  },
  backButton: {
    marginRight: 15,
    padding: 5,
  },
  backIcon: {
    fontSize: 28,
    color: '#FFFFFF',
    fontWeight: 'bold',
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#FFFFFF',
    flex: 1,
  },
  section: {
    backgroundColor: '#FFFFFF',
    marginVertical: 10,
    paddingVertical: 15,
    paddingHorizontal: 20,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#424242',
    marginBottom: 15,
  },
  settingRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#F0F0F0',
  },
  settingInfo: {
    flex: 1,
    marginRight: 15,
  },
  settingLabel: {
    fontSize: 16,
    fontWeight: '500',
    color: '#212121',
    marginBottom: 4,
  },
  settingDescription: {
    fontSize: 13,
    color: '#757575',
  },
  sliderContainer: {
    paddingVertical: 10,
  },
  sliderValue: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1E88E5',
    textAlign: 'center',
    marginTop: 10,
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#F0F0F0',
  },
  infoLabel: {
    fontSize: 15,
    color: '#616161',
  },
  infoValue: {
    fontSize: 15,
    fontWeight: '500',
    color: '#212121',
  },
  helpButton: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 15,
  },
  helpButtonText: {
    fontSize: 16,
    color: '#1E88E5',
    fontWeight: '500',
  },
  helpIcon: {
    fontSize: 20,
    color: '#1E88E5',
  },
});