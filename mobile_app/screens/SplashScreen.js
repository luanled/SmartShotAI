import React, { useEffect } from 'react';
import { View, Text, StyleSheet, ActivityIndicator } from 'react-native';

export default function SplashScreen({ navigation }) {
  useEffect(() => {
    const timer = setTimeout(() => {
      navigation.replace('LiveCamera');
    }, 3000);
    return () => clearTimeout(timer);
  }, [navigation]);

  return (
    <View style={styles.container}>
      <View style={styles.iconContainer}>
        <View style={styles.iconPlaceholder}>
          <Text style={styles.iconText}>📷</Text>
        </View>
      </View>
      <View style={styles.brandContainer}>
        <Text style={styles.brandText}>SmartShot AI</Text>
      </View>
      <View style={styles.statusContainer}>
        <Text style={styles.statusText}>Initializing app ...</Text>
        <Text style={styles.statusSubText}>Initializing aesthetic model ...</Text>
        <ActivityIndicator size="large" color="#1E88E5" style={styles.loader} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#D6E9F8',
    justifyContent: 'center',
    alignItems: 'center',
  },
  iconContainer: {
    marginBottom: 30,
  },
  iconPlaceholder: {
    width: 120,
    height: 120,
    backgroundColor: '#1E88E5',
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 4,
    borderColor: '#0D47A1',
  },
  iconText: {
    fontSize: 60,
  },
  brandContainer: {
    backgroundColor: '#1565C0',
    paddingHorizontal: 40,
    paddingVertical: 15,
    borderRadius: 8,
    marginBottom: 80,
  },
  brandText: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#FFFFFF',
  },
  statusContainer: {
    alignItems: 'center',
  },
  statusText: {
    fontSize: 16,
    color: '#424242',
    marginBottom: 5,
  },
  statusSubText: {
    fontSize: 14,
    color: '#616161',
    marginBottom: 20,
  },
  loader: {
    marginTop: 10,
  },
});