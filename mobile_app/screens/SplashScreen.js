import { useEffect } from 'react';
import { View, Text, ActivityIndicator } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import styles from '../styles/SplashScreen.styles';

export default function SplashScreen({ navigation }) {
  useEffect(() => {
    const timer = setTimeout(() => {
      navigation.replace('LiveCamera');
    }, 2500);
    return () => clearTimeout(timer);
  }, [navigation]);

  return (
    <LinearGradient colors={['#0A0E1A', '#0D1F3C', '#0A0E1A']} style={styles.container}>
      <View style={styles.iconGlow}>
        <View style={styles.iconCircle}>
          <Ionicons name="camera" size={62} color="#fff" />
        </View>
      </View>

      <View style={styles.brandRow}>
        <Text style={styles.brandText}>SmartShot</Text>
        <Text style={styles.brandAccent}> AI</Text>
      </View>
      <Text style={styles.tagline}>AI-Powered Photography Guidance</Text>

      <View style={styles.loaderArea}>
        <ActivityIndicator size="small" color="#3B82F6" />
        <Text style={styles.loaderText}>Initializing aesthetic model…</Text>
      </View>
    </LinearGradient>
  );
}
