import { useState } from 'react';
import { View, Text, Switch, ScrollView, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import styles from '../styles/SettingsScreen.styles';

export default function SettingsScreen({ navigation }) {
  const [voiceGuidance, setVoiceGuidance] = useState(true);
  const [vibrationFeedback, setVibrationFeedback] = useState(true);
  const [personalizationLevel] = useState(50);

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity style={styles.backButton} onPress={() => navigation.goBack()}>
          <Ionicons name="arrow-back" size={22} color="#F0F6FC" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Settings</Text>
      </View>

      <ScrollView style={styles.scroll} contentContainerStyle={styles.scrollContent}>
        {/* Feedback */}
        <Text style={styles.sectionLabel}>FEEDBACK</Text>
        <View style={styles.card}>
          <View style={styles.settingRow}>
            <View style={styles.settingLeft}>
              <View style={[styles.iconBox, { backgroundColor: 'rgba(59,130,246,0.15)' }]}>
                <Ionicons name="volume-high-outline" size={18} color="#3B82F6" />
              </View>
              <View>
                <Text style={styles.settingLabel}>Voice Guidance</Text>
                <Text style={styles.settingDesc}>Hear audio directions</Text>
              </View>
            </View>
            <Switch
              value={voiceGuidance}
              onValueChange={setVoiceGuidance}
              trackColor={{ false: '#30363D', true: '#1D4ED8' }}
              thumbColor={voiceGuidance ? '#3B82F6' : '#8B949E'}
            />
          </View>

          <View style={styles.divider} />

          <View style={styles.settingRow}>
            <View style={styles.settingLeft}>
              <View style={[styles.iconBox, { backgroundColor: 'rgba(168,85,247,0.15)' }]}>
                <Ionicons name="phone-portrait-outline" size={18} color="#A855F7" />
              </View>
              <View>
                <Text style={styles.settingLabel}>Vibration Feedback</Text>
                <Text style={styles.settingDesc}>Haptic feedback for guidance</Text>
              </View>
            </View>
            <Switch
              value={vibrationFeedback}
              onValueChange={setVibrationFeedback}
              trackColor={{ false: '#30363D', true: '#6D28D9' }}
              thumbColor={vibrationFeedback ? '#A855F7' : '#8B949E'}
            />
          </View>
        </View>

        {/* Personalization */}
        <Text style={styles.sectionLabel}>PERSONALIZATION</Text>
        <View style={styles.card}>
          <View style={styles.settingRow}>
            <View style={styles.settingLeft}>
              <View style={[styles.iconBox, { backgroundColor: 'rgba(34,197,94,0.15)' }]}>
                <Ionicons name="sparkles-outline" size={18} color="#22C55E" />
              </View>
              <View>
                <Text style={styles.settingLabel}>AI Adaptation Level</Text>
                <Text style={styles.settingDesc}>How much the AI adapts to your style</Text>
              </View>
            </View>
            <Text style={styles.settingValue}>{personalizationLevel}%</Text>
          </View>
        </View>

        {/* App Info */}
        <Text style={styles.sectionLabel}>APP INFO</Text>
        <View style={styles.card}>
          <View style={styles.infoRow}>
            <Text style={styles.infoLabel}>Model Version</Text>
            <Text style={styles.infoValue}>1.0.0</Text>
          </View>
          <View style={styles.divider} />
          <View style={styles.infoRow}>
            <Text style={styles.infoLabel}>Build</Text>
            <Text style={styles.infoValue}>01</Text>
          </View>
        </View>

        {/* Support */}
        <Text style={styles.sectionLabel}>SUPPORT</Text>
        <View style={styles.card}>
          <TouchableOpacity style={styles.settingRow} activeOpacity={0.7}>
            <View style={styles.settingLeft}>
              <View style={[styles.iconBox, { backgroundColor: 'rgba(249,115,22,0.15)' }]}>
                <Ionicons name="help-circle-outline" size={18} color="#F97316" />
              </View>
              <Text style={styles.settingLabel}>About / Help</Text>
            </View>
            <Ionicons name="chevron-forward" size={18} color="#8B949E" />
          </TouchableOpacity>
        </View>
      </ScrollView>
    </View>
  );
}
