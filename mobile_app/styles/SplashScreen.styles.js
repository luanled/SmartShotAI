import { StyleSheet } from 'react-native';
import { colors } from './theme';

export default StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  iconGlow: {
    width: 130,
    height: 130,
    borderRadius: 65,
    backgroundColor: 'rgba(59,130,246,0.12)',
    borderWidth: 1,
    borderColor: 'rgba(59,130,246,0.3)',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 32,
  },
  iconCircle: {
    width: 100,
    height: 100,
    borderRadius: 50,
    backgroundColor: colors.accent,
    justifyContent: 'center',
    alignItems: 'center',
  },
  brandRow: {
    flexDirection: 'row',
    alignItems: 'baseline',
    marginBottom: 8,
  },
  brandText: {
    fontSize: 34,
    fontWeight: '700',
    color: colors.textPrimary,
    letterSpacing: 0.5,
  },
  brandAccent: {
    fontSize: 34,
    fontWeight: '700',
    color: colors.accentLight,
    letterSpacing: 0.5,
  },
  tagline: {
    fontSize: 13,
    color: colors.textSecondary,
    letterSpacing: 1,
    marginBottom: 80,
  },
  loaderArea: {
    position: 'absolute',
    bottom: 60,
    flexDirection: 'row',
    alignItems: 'center',
  },
  loaderText: {
    fontSize: 13,
    color: colors.textSecondary,
    marginLeft: 10,
  },
});
