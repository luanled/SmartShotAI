import { View, Text, ScrollView, TouchableOpacity } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import styles from '../styles/SettingsScreen.styles';
import { colors, radius } from '../styles/theme';
import { useModel } from '../context/ModelContext';

const MODEL_OPTIONS = [
  {
    key: 'optimized',
    label: 'Optimized  7 MB',
    desc: 'Faster inference, smaller size',
    icon: 'flash-outline',
    color: '#22C55E',
  },
  {
    key: 'standard',
    label: 'Standard  20 MB',
    desc: 'Full-precision weights',
    icon: 'layers-outline',
    color: '#3B82F6',
  },
  {
    key: 'siglip',
    label: 'SigLIP  356 MB',
    desc: 'Vision-language aesthetic model',
    icon: 'eye-outline',
    color: '#A855F7',
  },
];

export default function SettingsScreen({ navigation }) {
  const { selectedModel, setSelectedModel } = useModel();

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

        {/* Model Selection */}
        <Text style={styles.sectionLabel}>SCORING MODEL</Text>
        <View style={styles.card}>
          {MODEL_OPTIONS.map((opt, idx) => {
            const selected = selectedModel === opt.key;
            return (
              <View key={opt.key}>
                {idx > 0 && <View style={styles.divider} />}
                <TouchableOpacity
                  style={styles.settingRow}
                  onPress={() => setSelectedModel(opt.key)}
                  activeOpacity={0.7}
                >
                  <View style={styles.settingLeft}>
                    <View style={[styles.iconBox, { backgroundColor: selected ? `${opt.color}22` : 'rgba(139,148,158,0.1)' }]}>
                      <Ionicons name={opt.icon} size={18} color={selected ? opt.color : colors.textSecondary} />
                    </View>
                    <View>
                      <Text style={[styles.settingLabel, selected && { color: opt.color }]}>{opt.label}</Text>
                      <Text style={styles.settingDesc}>{opt.desc}</Text>
                    </View>
                  </View>
                  {selected && <Ionicons name="checkmark-circle" size={20} color={opt.color} />}
                </TouchableOpacity>
              </View>
            );
          })}
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
