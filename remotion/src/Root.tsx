import { Composition } from 'remotion';
import { ArchitectureVideo } from './compositions/ArchitectureVideo';
import { TOTAL_DURATION, FPS } from './utils/timings';

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="ArchitectureVideo"
        component={ArchitectureVideo}
        durationInFrames={TOTAL_DURATION}
        fps={FPS}
        width={1920}
        height={1080}
        defaultProps={{}}
      />
      {/* Preview compositions for individual scenes */}
      <Composition
        id="LogoIntro"
        component={ArchitectureVideo}
        durationInFrames={240}
        fps={FPS}
        width={1920}
        height={1080}
        defaultProps={{ previewScene: 'logo' }}
      />
      <Composition
        id="Dropdown"
        component={ArchitectureVideo}
        durationInFrames={510}
        fps={FPS}
        width={1920}
        height={1080}
        defaultProps={{ previewScene: 'dropdown' }}
      />
    </>
  );
};
