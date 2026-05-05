import { StyleSheet } from 'react-native';
import { colors, radius } from './theme';

export default StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.bg,
  },
  photoArea: {
    flex: 1,
    padding: 16,
  },
  photoPlaceholder: {
    flex: 1,
    backgroundColor: colors.surface,
    borderRadius: radius.xl,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: colors.divider,
  },
  photo: {
    width: '100%',
    height: '100%',
    borderRadius: radius.xl,
  },
  photoLabel: {
    fontSize: 16,
    color: colors.textSecondary,
    marginTop: 12,
    marginBottom: 12,
  },
  scorePill: {
    position: 'absolute',
    bottom: 16,
    paddingHorizontal: 14,
    paddingVertical: 5,
    borderRadius: radius.round,
    borderWidth: 1.5,
    backgroundColor: 'rgba(13,17,23,0.75)',
  },
  scorePillText: {
    fontSize: 14,
    fontWeight: '600',
  },
  sheet: {
    backgroundColor: colors.surface,
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    paddingHorizontal: 20,
    paddingTop: 16,
    paddingBottom: 20,
    borderTopWidth: 1,
    borderTopColor: colors.divider,
  },
  sheetTitle: {
    fontSize: 17,
    fontWeight: '600',
    color: colors.textPrimary,
    textAlign: 'center',
    marginBottom: 20,
  },
  thumbsRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 28,
  },
  thumbBtn: {
    alignItems: 'center',
    paddingVertical: 16,
    paddingHorizontal: 28,
    borderRadius: radius.lg,
    backgroundColor: colors.card,
    borderWidth: 1.5,
    borderColor: 'transparent',
  },
  thumbBtnPositive: {
    borderColor: colors.accent,
    backgroundColor: colors.accentDim,
  },
  thumbBtnNegative: {
    borderColor: colors.danger,
    backgroundColor: colors.dangerDim,
  },
  thumbLabel: {
    fontSize: 13,
    color: colors.textSecondary,
    marginTop: 6,
    fontWeight: '500',
  },
  thumbLabelPositive: {
    color: colors.accent,
  },
  thumbLabelNegative: {
    color: colors.danger,
  },
  actionRow: {
    flexDirection: 'row',
    gap: 12,
  },
  discardBtn: {
    flex: 1,
    paddingVertical: 11,
    borderRadius: radius.md,
    borderWidth: 1.5,
    borderColor: colors.border,
    alignItems: 'center',
    justifyContent: 'center',
  },
  discardBtnText: {
    fontSize: 15,
    fontWeight: '600',
    color: colors.textSecondary,
  },
  saveBtn: {
    flex: 2,
    paddingVertical: 11,
    borderRadius: radius.md,
    backgroundColor: colors.accent,
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
  },
  saveBtnText: {
    fontSize: 15,
    fontWeight: '600',
    color: colors.white,
  },
});
