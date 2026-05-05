const { withAppBuildGradle, withDangerousMod } = require('@expo/config-plugins');
const path = require('path');
const fs = require('fs');

const ANDROID_DEP = 'org.tensorflow:tensorflow-lite-select-tf-ops';
const IOS_POD = "pod 'TensorFlowLiteSelectTfOps'";

const withAndroidFlexDelegate = (config) =>
  withAppBuildGradle(config, (cfg) => {
    if (!cfg.modResults.contents.includes(ANDROID_DEP)) {
      cfg.modResults.contents = cfg.modResults.contents.replace(
        /dependencies\s*\{/,
        `dependencies {\n    implementation "${ANDROID_DEP}:+"`,
      );
    }
    return cfg;
  });

const withIosFlexDelegate = (config) =>
  withDangerousMod(config, [
    'ios',
    async (cfg) => {
      const podfilePath = path.join(cfg.modRequest.platformProjectRoot, 'Podfile');
      let contents = fs.readFileSync(podfilePath, 'utf-8');
      if (!contents.includes('TensorFlowLiteSelectTfOps')) {
        contents = contents.replace(
          /use_expo_modules!/,
          `${IOS_POD}\n  use_expo_modules!`,
        );
        fs.writeFileSync(podfilePath, contents);
      }
      return cfg;
    },
  ]);

module.exports = (config) => {
  config = withAndroidFlexDelegate(config);
  config = withIosFlexDelegate(config);
  return config;
};
