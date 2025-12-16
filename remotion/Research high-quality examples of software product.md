<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Research high-quality examples of software product demo and architecture explainer motion graphics.

Focus on:
Frame-by-frame breakdowns of product demo videos
Motion structure for SaaS, developer tools, and AI platforms
Scene sequencing (intro, problem, architecture, features, CTA)
How UI elements, diagrams, and data flows are animated
Timing conventions (seconds per scene, transitions, emphasis)
Scriptwriting structure aligned with visuals
Provide:
Example scripts mapped to visual frames or timestamps
Storyboard-style explanations usable for Remotion implementation
Best practices used by top SaaS companies (Stripe, Vercel, Figma, OpenAI-style demos)
References to breakdowns, blogs, or public design analyses

Perfect. Now I have extensive research. Let me create a comprehensive guide with concrete examples and best practices.

## Software Product Demo \& Architecture Explainer Motion Graphics: Complete Guide

High-quality product demos and architecture explainers require a carefully orchestrated balance of **visual clarity, strategic pacing, and technical precision**. The following breakdown synthesizes best practices from companies like Stripe, Vercel, Figma, and leading SaaS platforms, providing you with actionable frameworks for Remotion implementation.

***

## Scene Sequencing \& Structural Framework

### Standard Demo Arc (60-90 seconds)

The optimal SaaS product demo follows this distribution:[^1][^2][^3]


| **Section** | **Duration** | **Frame Count (30fps)** | **Purpose** |
| :-- | :-- | :-- | :-- |
| **Hook** | 0:00–0:15 | 450 | State the problem, stop the scroll |
| **Problem** | 0:15–0:45 | 900 | Agitate the pain, build empathy |
| **Solution Overview** | 0:45–1:30 | 1,350 | Show core workflow in action |
| **Key Features** | 1:30–3:00 | 4,500 | Demonstrate 3-4 power features |
| **Social Proof** | 3:00–3:30 | 900 | Results from real customers |
| **CTA** | 3:30–4:00 | 900 | Clear, actionable next step |

For **longer, mid-funnel demos** (2-5 minutes), each feature section can expand to 30-45 seconds, providing deeper workflow context.[^2][^4]

***

## Timing Conventions \& Pacing Standards

### Voice-Over Alignment

**Optimal speaking pace: 150-160 words per minute**[^5]

- **60-second demo:** ~150 words
- **90-second demo:** ~225-240 words
- **3-minute demo:** ~450-480 words

This matches the viewer's cognitive load—fast enough to maintain engagement, slow enough for comprehension of technical concepts.

### Scene Duration Principles

**Animation scene duration should follow these conventions:**[^6][^7][^8][^9]


| **Animation Type** | **Typical Duration** | **Frame Timing** | **Use Case** |
| :-- | :-- | :-- | :-- |
| **UI fade-in** | 300–500ms | 9–15 frames (30fps) | Element introduction |
| **Icon animation** | 600–900ms | 18–27 frames | Emphasis marker |
| **Scroll/pan** | 800–1,200ms | 24–36 frames | Scene transition |
| **Data visualization build** | 1.5–2.5s | 45–75 frames | Graph, chart reveal |
| **Feature demo loop** | 3–5s | 90–150 frames | Full workflow cycle |

**Rule:** Slower animations (700ms+) feel more professional and intentional. Faster micro-animations (300ms) should be reserved for emphasis and UI feedback.[^7]

***

## Easing \& Motion Curves for Professional Feel

### Easing Strategy by Animation Type

**Ease-in-out (default):**[^10][^11][^12]

- Starts slower, accelerates, then decelerates at end
- Feels most natural and deliberate
- Best for general UI transitions and feature reveals
- Typical duration: 600–1,200ms

**Ease-out (recommended for entrances):**[^11]

- Starts at peak velocity, slows at end
- Feels responsive and snappy
- Perfect for callout boxes, highlight animations
- Typical duration: 400–700ms

**Ease-in (recommended for exits):**[^11]

- Accelerates gradually then stops
- Conveys weight and finality
- Good for elements leaving the frame
- Typical duration: 500–800ms

**Spring (physics-based, for emphasis):**[^10][^11]

- Creates subtle bounce and liveliness
- Used for cursor clicks, icon reveals, or celebratory moments
- Typical bounce: 0.5–1.0 stiffness
- Duration determined by physics, not milliseconds

***

## Frame-by-Frame Storyboard Structure

### Example: 90-Second SaaS Feature Demo (Workflow Automation Tool)

**Scene 1: Problem Hook (0:00–0:10)**

```
Visual: Frustrated user at cluttered desk, multiple windows open
VO: "Switching between apps wastes 4+ hours per week"
Motion: Zoom-in on user's face (0–0.5s, ease-out)
        Pulse effect on open tabs (0.3–0.7s, ease-in-out)
Notes: Use warm, saturated color palette for problem context
```

**Scene 2: Product Introduction (0:10–0:25)**

```
Visual: Clean dashboard appears in center
VO: "Meet [Product]: All your workflows in one place"
Motion: Fade in background (0–0.3s, ease-out)
        Scale product logo from 80% → 100% (0.3–0.8s, ease-out)
        Slide title in from left (0.5–0.9s, ease-out)
Notes: Use brand blue. Ensure 20px padding around key elements.
```

**Scene 3: Core Feature Walkthrough (0:25–1:00)**

```
Visual: Dashboard with task list, drag-and-drop in action
VO: "Drag tasks to automate your workflow. No coding required."
Motion: 
  - Cursor appears (0s, instant)
  - Cursor bounces on task item (0–0.3s, spring physics)
  - Task drags to automation column (0.5–1.2s, ease-out, 3D perspective)
  - Checkmark appears (1.2–1.5s, scale + fade-in)
  - Automation badge pulses (1.5–2.0s, ease-in-out)
Notes: Motion tracking follows cursor throughout. Highlight destination
       area with semi-transparent background.
```

**Scene 4: Results/Proof (1:00–1:20)**

```
Visual: Before/after metrics side-by-side
VO: "Save 8+ hours weekly. Acme Inc. reduced errors by 60%."
Motion: 
  - Left side (before state) slides in from left (0–0.6s, ease-out)
  - Right side (after state) slides in from right (0.2–0.8s, ease-out)
  - Numbers count up: 8h → animated number (0.8–2.0s, linear)
  - Green checkmark fade-in (1.8–2.2s, scale + ease-out)
Notes: Use animated SVG for number counters. Brand green for positive results.
```

**Scene 5: CTA (1:20–1:30)**

```
Visual: Button slides up from bottom, glow effect
VO: "Start your free trial today."
Motion: 
  - Button background fade-in (0–0.3s, ease-out)
  - Text fade-in (0.1–0.4s, ease-out)
  - Glow pulse (0.5–1.0s, ease-in-out, repeat 2x)
Notes: Button hover state not shown in video. Use contrasting CTA color.
```


***

## Visual Hierarchy \& Emphasis Techniques

### Strategic Attention Direction[^7]

1. **Size \& Scale:**
    - Key elements: increase size by 15–25%
    - Supporting info: maintain original size or scale down 10–15%
    - Use scale transitions (300–600ms) for smooth emphasis
2. **Color Contrast:**
    - Highlight critical UI: use 10–15% lighter/darker shade or complementary color
    - De-emphasize background: reduce opacity to 40–60%
    - Use brand accent color for CTA elements
3. **Animation Timing (Sequential Reveals):**

```
0–15 seconds: Introduce problem (1 visual element)
15–30 seconds: Show solution (2–3 elements, staggered by 200–300ms)
30–45 seconds: Feature detail (3–4 elements, each 400–600ms apart)
45–60 seconds: Results (animated data, 1.5–2.5s duration)
```

4. **Signifiers (Motion Cues):**[^7]
    - **Arrows \& pointers:** Animate entrance (ease-out, 400–600ms), dwell (1–2s), exit (ease-in, 300–500ms)
    - **Highlight boxes:** Fade in (300ms, ease-out), glow pulse (1.2s, ease-in-out), fade out (300ms, ease-in)
    - **Cursor emphasis:** Bounce on click area (spring, 0.5–1.0s), then smooth trace path
    - **Progress indicators:** Build left-to-right (800ms–1.5s, linear or ease-out)

***

## UI Element Animation Techniques for Demo Videos

### Screen Recording Motion Graphics Overlay Strategy

**Cursor Animation Best Practices:**[^13][^14]

- **Cursor visibility:** Hide during 2+ second pauses; show for active demonstration
- **Cursor smoothing:** Apply 100–200ms Bézier smoothing for natural movement
- **Click effects:** Pulse + glow (300–500ms, ease-out) on mousedown
- **Available styles:** 40+ cursor options; 8 click effect variations (shadow, glow, ripple, etc.)

**Highlighting \& Callout Annotations:**

- **Animated boxes:** Fade in (300ms), pulse glow (600–800ms), fade out
- **Text callouts:** Slide in from edge (400–700ms, ease-out), hold 2–3s, slide out
- **Arrow indicators:** Draw via SVG path animation (600–1000ms, linear stroke-dashoffset)
- **Zoom effect:** Pan to highlighted area (800–1200ms, ease-out), zoom 1.2–1.5x

**Structured Emphasis Sequence for a Single Feature:**

```
T=0s:     Cursor moves to feature area
T=0.5s:   Destination box appears (semi-transparent, ease-out)
T=0.8s:   Arrow pointer animates in, pointing to feature
T=1.2s:   Text callout slides in from left
T=2.5s:   Cursor clicks (spring animation)
T=2.8s:   Destination box fades; next area highlights
T=3.5s:   Callout fades; narration continues
```


***

## Architecture \& Data Flow Animation

### Animated Diagram Principles[^15][^16]

**System Architecture Animation Structure:**

1. **Component Introduction Phase (0–1.5s):**
    - Fade in static components at staggered intervals (200–300ms apart)
    - Use entrance easing (ease-out, 400–600ms per component)
2. **Connection Phase (1.5–3.0s):**
    - Draw animated connection lines (SVG stroke-dasharray animation, linear, 800–1200ms)
    - Use path reveal with trailing glow effect (optional, 300ms offset)
3. **Data Flow Phase (3.0–5.0s):**
    - Animate data packets moving along paths (ease-out-back, 1.2–1.8s)
    - Color-code data types (API requests = blue, responses = green, errors = red)
    - Stagger multiple packet flows by 400–600ms for rhythm
4. **Result/Summary Phase (5.0+):**
    - Highlight key metrics or results
    - Pulse important nodes (ease-in-out, 1.2–1.5s, 2–3 cycles)

**Example: Microservice Architecture Demo**

```
T=0–0.6s:   API Gateway fades in
T=0.3–0.9s: Service A fades in (staggered)
T=0.6–1.2s: Service B fades in
T=0.9–1.5s: Database fades in
T=1.5–2.5s: API→Service A connection line draws (linear, 1s)
T=1.8–2.8s: Service A→DB connection draws
T=2.0–3.2s: Service B→Cache connection draws
T=3.0–5.0s: Data packets animate: Request→ServiceA→DB (staggered by 0.4s each)
T=5.0–6.0s: Success state: all nodes glow green (ease-in-out pulse)
```


***

## Script-to-Visual Mapping

### Template: 90-Second Demo Script with Timings

```
[SCENE 1: PROBLEM] (0:00–0:15)
VO: "Every day, your team wastes hours juggling spreadsheets 
     and emails just to track a single project."
Visual: [T=0s] Frustrated user at desk, multiple windows
        [T=0.3s] Tab switch animation (flashing tabs)
        [T=0.8s] Zoom on overwhelming dashboard
        [T=1.5s] Close-up: user's frustrated expression
        [Duration: 15 frames × 2 = 30 frames total]

[SCENE 2: SOLUTION INTRO] (0:15–0:30)
VO: "Introducing [Product]. All your projects in one place."
Visual: [T=0s] Fade to clean, minimal dashboard
        [T=0.3s] Product logo scales in (ease-out, 600ms)
        [T=0.8s] Main headline slides in from left (ease-out, 500ms)
        [Duration: 450 frames]

[SCENE 3: FEATURE DEMO] (0:30–1:00)
VO: "Drag any task into your workflow. That's it. No setup needed."
Visual: [T=0s] Cursor appears over task item
        [T=0.2s] Task item highlights (background color shift)
        [T=0.5s] Cursor begins drag motion (ease-in-out, 1.0s)
        [T=1.2s] Drop zone highlights
        [T=1.5s] Task animates to new location (spring, 0.8s)
        [T=2.0s] Green checkmark appears (scale + fade, 600ms)
        [T=3.0s] Automation status updates live
        [Duration: 900 frames]

[SCENE 4: PROOF] (1:00–1:20)
VO: "Acme reduced project delays by 40%. Save 8 hours weekly."
Visual: [T=0s] Two-column comparison fades in
        [T=0.4s] "Before" metrics slide in from left (ease-out, 700ms)
        [T=0.7s] "After" metrics slide in from right (ease-out, 700ms)
        [T=1.2s] Number counter: "40%" animates (linear 0–40, 1.5s)
        [T=2.5s] Green checkmark pulses (ease-in-out, 1.0s, 2 cycles)
        [Duration: 600 frames]

[SCENE 5: CTA] (1:20–1:30)
VO: "Start your free trial today. No credit card required."
Visual: [T=0s] CTA button slides up from bottom (ease-out, 600ms)
        [T=0.2s] Button text fades in
        [T=0.6s] Glow effect pulses (ease-in-out, 1.0s, repeat 2x)
        [Duration: 300 frames]
```


***

## Remotion Implementation Patterns

### Timing Architecture for 30fps Videos

```javascript
// Standard scene timing (frames at 30fps)
const SCENE_DURATIONS = {
  hook: 450,              // 0:00–0:15
  problem: 900,           // 0:15–0:45
  solutionOverview: 1350, // 0:45–1:30
  featureDemo: 4500,      // 1:30–3:00
  proof: 900,             // 3:00–3:30
  cta: 900,               // 3:30–4:00
};

// Animation keyframe reference (in frames)
const ANIMATIONS = {
  fadeInQuick: { duration: 9, easing: 'easeOut' },      // 300ms
  fadeInMedium: { duration: 18, easing: 'easeOut' },    // 600ms
  slideInMedium: { duration: 21, easing: 'easeOut' },   // 700ms
  pulseGlow: { duration: 36, easing: 'easeInOut' },     // 1.2s
  dataVisBuild: { duration: 45, easing: 'easeOut' },    // 1.5s
};

// Frame-accurate timing utility
const getFrameTime = (timeInSeconds) => timeInSeconds * 30;
```


### Easing Function Reference

```javascript
// Recommended easing functions for Remotion
const easingFunctions = {
  easeInOut: (t) => t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t,
  easeOut: (t) => t * (2 - t),
  easeIn: (t) => t * t,
  easeOutBack: (t) => 1 + (--t) * t * (2.70158 * t + 1.70158),
  linearSpring: (t) => t, // For data counters
};

// Example: Framer Motion easing in Remotion context
const cursorBounce = {
  type: "spring",
  stiffness: 100,
  damping: 10,
  duration: 300,
};
```


***

## Best Practices from Industry Leaders

### Stripe's Approach[^17][^18][^19]

Stripe uses a **layered motion graphics strategy:**

- **Static design phase:** Build complete visual in Figma first
- **Staggered component animation:** Each UI element enters at slightly different times (200–300ms offset)
- **Unified timing:** All elements arrive at the same final endpoint despite different speeds
- **Smooth curves:** Use 90% keyframe velocity influence for natural deceleration
- **GPU acceleration:** Enable `will-change` and `transform` animations (not layout shifts)


### Vercel's Animation Philosophy[^20][^21]

Vercel emphasizes **interruptible, responsive animations:**

- Animations can be interrupted mid-flight without jarring jumps
- Uses Framer Motion for declarative animation control
- Focuses on **shared layout animations** for page transitions
- Avoids excessive motion; animation serves UX purpose, not aesthetics
- Tests on low-end devices to ensure smooth 60fps performance


### Figma's UI Animation Strategy[^22][^23]

Figma combines **screencast with fluid animations:**

- Real-time UI captures with motion graphics overlays
- "Clicky" sound effects synced to cursor interactions
- Smooth 2D character movement for narrative flows
- Smart Animate feature for seamless frame-to-frame transitions
- Emphasis on workflow clarity over visual spectacle

***

## Scriptwriting Structure Aligned with Motion

### The PAS Framework (Problem-Agitation-Solution)[^24]

**Problem (0:00–0:15):**

- State a pain point your audience experiences
- Keep visual minimal; focus on emotional connection
- Narration: 1–2 sentences max

**Agitation (0:15–0:45):**

- Show consequences or broader impact of problem
- Use visual metaphors (clutter, chaos, time slipping away)
- Narration: 2–3 sentences with emotional weight

**Solution (0:45–4:00):**

- Introduce product as "the answer"
- Walkthrough features that directly address pain points
- Show real-world results and proof
- Narration: Clear, action-oriented

**Call-to-Action (3:30–4:00):**

- Direct, specific instruction: "Start free trial," "Watch demo," "Schedule call"
- Avoid passive language; use imperative mood
- Ensure alignment between button on screen and voiceover

***

## Advanced Techniques: Data Visualization \& Animation

### Animated Number Counters[^25][^24]

```javascript
// Animate a number from 0 to target value over duration
const AnimatedCounter = ({ 
  from = 0, 
  to = 100, 
  duration = 1500, // ms
  format = (n) => n 
}) => {
  return (
    <motion.div
      initial={{ count: from }}
      animate={{ count: to }}
      transition={{ duration: duration / 1000, ease: "easeOut" }}
    >
      {/* Use Framer Motion `useMotionValue` + `useTransform` */}
    </motion.div>
  );
};
```


### SVG Path Reveal (Data Flows, Connections)[^16][^26]

```javascript
// Animate SVG path stroke for data flow visualization
const AnimatedPath = ({ 
  pathLength = 100, 
  duration = 1200 // ms
}) => (
  <motion.path
    initial={{ strokeDashoffset: pathLength }}
    animate={{ strokeDashoffset: 0 }}
    transition={{ 
      duration: duration / 1000, 
      ease: "easeOut" 
    }}
    style={{ 
      strokeDasharray: pathLength,
      fill: 'none',
      stroke: '#0066FF',
      strokeWidth: 2
    }}
  />
);
```


***

## Video Length \& Audience Awareness

| **Video Length** | **Awareness Level** | **Best Use** | **Content Scope** |
| :-- | :-- | :-- | :-- |
| **30–60s** | Low (cold audiences) | Social media, top-of-funnel ads | 1 core benefit only |
| **60–90s** | Low-Medium (warm audiences) | Landing page hero, YouTube pre-roll | Problem + solution + 1 feature |
| **2–3m** | Medium (engaged prospects) | Sales deck, email campaign | Problem + 3–4 features + proof |
| **3–5m** | High (self-qualified leads) | Onboarding, detailed feature walkthrough | Complete workflow + customization options |
| **5m+** | Very High (qualified buyers) | In-depth training, technical deep-dives | Multiple workflows, advanced features, edge cases |


***

## Storyboard Checklist Before Animation

✅ **Script is final or 90% locked**
✅ **Each scene has visual reference** (rough sketch, UI screenshot, or Figma wireframe)
✅ **Voiceover/text is written** with estimated duration
✅ **Animation notes are explicit** ("fade in over 600ms," "zoom 1.5x," "ease-out")
✅ **Transitions between scenes are mapped** (fade, slide, zoom, cut)
✅ **Timing aligns with voiceover cadence**
✅ **Stakeholders have reviewed** and approved direction

***

## Tools \& Technologies for Implementation

**Remotion-Friendly Stack:**

- **Remotion** (core video engine)
- **Framer Motion** (easing, springs, component animations)
- **SVG.js** or **Three.js** (for 3D architecture diagrams)
- **React Number Format** (animated counters)
- **Canvas/Konva.js** (live data visualization)

**Complementary Screen Recording Tools:**

- **Loom** (quick screen + webcam, built-in cursor emphasis, 2–5min limits)[^14]
- **Supademo** (auto-zoom, smart annotations, emphasis tools)[^27]
- **FocuSee** (AI cursor animation, auto-zoom detection)[^13]

***

## Summary \& Quick Reference

**For your Remotion implementation, prioritize:**

1. **Scene Duration:** 30–45 seconds per feature; 60–90 seconds total for top-funnel
2. **Animation Easing:** Default to ease-out (400–700ms) for UI elements, ease-in-out (600–1200ms) for transitions
3. **Timing Offset:** Stagger related animations by 200–300ms for visual rhythm
4. **Visual Hierarchy:** Use size (15% scale change), color (10–15% brightness shift), and opacity (40–60% for de-emphasis)
5. **Voiceover Sync:** 150–160 words/minute; plan motion to match script cadence
6. **Cursor Behavior:** Hide during pauses; smooth movement (100–200ms interpolation); bounce on clicks (spring, 0.5–1.0s)
7. **Data Visualization:** Build charts/graphs over 1.5–2.5 seconds with linear or ease-out; animate counters with linear easing
8. **Storyboard First:** Lock visuals before recording voiceover; expect 10-minute storyboard approval saves 5–10 hours of rework

***

## References \& Case Studies

This guide synthesizes best practices from:[^28][^3][^29][^18][^30][^31][^21][^19][^12][^1][^2][^17][^22][^15][^24][^5][^14][^10][^11][^7]

For deeper dives, explore Stripe's developer video production methodology, Vercel's animation principles via Emil Kowalski's design notes, and Figma's prototyping documentation for advanced Smart Animate techniques.
<span style="display:none">[^32][^33][^34][^35][^36][^37][^38][^39][^40][^41][^42][^43][^44][^45][^46]</span>

<div align="center">⁂</div>

[^1]: https://advids.co/blog/saas-animated-tutorial

[^2]: https://vidico.com/news/video-marketing-case-study/

[^3]: https://videoai.me/blog/saas-product-demo-videos-conversion-format

[^4]: https://www.synthesia.io/post/product-demo-video-examples

[^5]: https://speechify.com/blog/create-voice-over-animated-video/

[^6]: https://www.youtube.com/watch?v=7ai-VuJ6YCo

[^7]: https://blog.dataiker.com/mastering-visual-hierarchy-in-video-production/

[^8]: https://www.youtube.com/watch?v=6UXjRCORV44

[^9]: https://www.clipstudio.net/how-to-draw/archives/172581

[^10]: https://www.framer.com/academy/lessons/framer-animations-transitions-and-easing

[^11]: https://www.youtube.com/watch?v=E6Vwh3ou5Ew

[^12]: https://www.youtube.com/watch?v=bOgUXDibygM

[^13]: https://focusee.imobie.com/features/auto-zoom-and-cursor-animation.htm

[^14]: https://www.flowjam.com/blog/how-to-create-a-loom-demo-the-complete-guide-for-professional-video-demos

[^15]: https://dev.to/aws-builders/spicing-up-aws-architecture-diagrams-a-step-by-step-guide-to-creating-animated-aws-architecture-gifs-jjb

[^16]: https://icraft.design/blog/animation

[^17]: https://mypromovideos.com/video-inspirations/video/payments-updates-innovations-motion-graphics-explainer-stripe/

[^18]: https://developerrelations.com/talks/stripe-multiplexed-video-production/

[^19]: https://www.youtube.com/watch?v=MT7vDKuTXe8

[^20]: https://emilkowal.ski/ui/great-animations

[^21]: https://vercel.com/blog/building-a-fast-animated-image-gallery-with-next-js

[^22]: https://www.moonb.io/blog/software-demo-script-examples

[^23]: https://www.youtube.com/watch?v=HCD8-JqCtH0

[^24]: https://www.moonb.io/blog/explainer-video-script

[^25]: https://www.contentbeta.com/blog/storyboard-for-animated-explainer-videos/

[^26]: https://svgenius.design/blog/real-time-data-feeds-driving-live-svg-animations

[^27]: https://supademo.com/blog/demonstration-video-examples/

[^28]: https://educationalvoice.co.uk/saas-explainer-video/

[^29]: https://motioncue.com/saas-explainer-videos-2025/

[^30]: https://www.sitepoint.com/remotion-create-animated-videos-using-html-css-react/

[^31]: https://www.youtube.com/watch?v=cGi2CcEJjRk

[^32]: https://www.contentbeta.com/blog/animated-explainer-company/

[^33]: https://explainvisually.co/en/software-explainer-videos/

[^34]: https://www.figma.com/resource-library/how-to-create-a-ux-storyboard/

[^35]: https://www.youtube.com/watch?v=VqJjBiC-atE

[^36]: https://www.contentbeta.com/blog/product-demo-scripts/

[^37]: https://project-aeon.com/blogs/how-to-overlay-videos-quick-pro-tips

[^38]: https://www.smashingmagazine.com/2022/06/precise-timing-web-animations-api/

[^39]: https://www.youtube.com/watch?v=zclrBaeCdpA

[^40]: https://developer.mozilla.org/en-US/docs/Web/API/Web_Animations_API/Using_the_Web_Animations_API

[^41]: https://www.youtube.com/watch?v=St_gAln5w4s

[^42]: https://www.truefan.ai/blogs/ai-product-demo-video-creator

[^43]: https://stackoverflow.com/questions/45975275/detecting-mouse-movement-in-a-screen-recording-using-computer-vision

[^44]: https://www.contentbeta.com/blog/motion-graphics-styles/

[^45]: https://blog.roboflow.com/video-annotation/

[^46]: https://nextjs.org/showcase

