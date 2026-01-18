"use client";

import { useState } from "react";
import { motion } from "motion/react";
import {
  User,
  Bell,
  Shield,
  CreditCard,
  Globe,
  Palette,
  Volume2,
  Save,
  Check,
  type LucideIcon
} from "lucide-react";

interface SettingsSection {
  id: string;
  title: string;
  icon: LucideIcon;
}

const sections: SettingsSection[] = [
  { id: "profile", title: "Profile", icon: User },
  { id: "notifications", title: "Notifications", icon: Bell },
  { id: "preferences", title: "Preferences", icon: Palette },
  { id: "security", title: "Security", icon: Shield },
  { id: "billing", title: "Billing", icon: CreditCard },
];

function ToggleSwitch({
  enabled,
  onChange
}: {
  enabled: boolean;
  onChange: (value: boolean) => void;
}) {
  return (
    <button
      onClick={() => onChange(!enabled)}
      className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
        enabled ? 'bg-blue-500' : 'bg-white/20'
      }`}
    >
      <span
        className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
          enabled ? 'translate-x-6' : 'translate-x-1'
        }`}
      />
    </button>
  );
}

export default function SettingsPage() {
  const [activeSection, setActiveSection] = useState("profile");
  const [saved, setSaved] = useState(false);

  // Settings state
  const [settings, setSettings] = useState({
    name: "John Doe",
    email: "john@company.com",
    language: "en",
    theme: "dark",
    emailNotifications: true,
    pushNotifications: false,
    soundEnabled: true,
    autoSave: true,
  });

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div className="max-w-5xl mx-auto px-6 py-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-3xl font-bold text-white mb-2">Settings</h1>
        <p className="text-slate-400">Manage your account preferences and settings.</p>
      </motion.div>

      <div className="flex gap-8">
        {/* Sidebar */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="w-64 flex-shrink-0"
        >
          <nav className="space-y-1">
            {sections.map((section) => {
              const Icon = section.icon;
              const isActive = activeSection === section.id;

              return (
                <button
                  key={section.id}
                  onClick={() => setActiveSection(section.id)}
                  className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-left transition-colors ${
                    isActive
                      ? 'bg-gradient-to-r from-blue-500/20 to-purple-500/20 text-white border border-blue-500/30'
                      : 'text-slate-400 hover:bg-white/5 hover:text-white'
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  {section.title}
                </button>
              );
            })}
          </nav>
        </motion.div>

        {/* Content */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="flex-1"
        >
          <div className="bg-white/5 border border-white/10 rounded-2xl p-6">
            {/* Profile Section */}
            {activeSection === "profile" && (
              <div className="space-y-6">
                <h2 className="text-xl font-semibold text-white mb-6">Profile Settings</h2>

                <div className="flex items-center gap-6 mb-8">
                  <div className="w-20 h-20 rounded-full bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center text-white text-2xl font-bold">
                    JD
                  </div>
                  <div>
                    <button className="px-4 py-2 bg-white/10 rounded-lg text-white text-sm hover:bg-white/20 transition-colors">
                      Change Avatar
                    </button>
                  </div>
                </div>

                <div className="grid gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">Full Name</label>
                    <input
                      type="text"
                      value={settings.name}
                      onChange={(e) => setSettings({ ...settings, name: e.target.value })}
                      className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">Email Address</label>
                    <input
                      type="email"
                      value={settings.email}
                      onChange={(e) => setSettings({ ...settings, email: e.target.value })}
                      className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                    />
                  </div>
                </div>
              </div>
            )}

            {/* Preferences Section */}
            {activeSection === "preferences" && (
              <div className="space-y-6">
                <h2 className="text-xl font-semibold text-white mb-6">Preferences</h2>

                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      <div className="flex items-center gap-2">
                        <Globe className="w-4 h-4" />
                        Language
                      </div>
                    </label>
                    <select
                      value={settings.language}
                      onChange={(e) => setSettings({ ...settings, language: e.target.value })}
                      className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                    >
                      <option value="en">English</option>
                      <option value="fr">French</option>
                    </select>
                  </div>

                  <div className="flex items-center justify-between py-4 border-b border-white/10">
                    <div className="flex items-center gap-3">
                      <Volume2 className="w-5 h-5 text-slate-400" />
                      <div>
                        <p className="text-white font-medium">Sound Effects</p>
                        <p className="text-slate-400 text-sm">Play sounds during conversations</p>
                      </div>
                    </div>
                    <ToggleSwitch
                      enabled={settings.soundEnabled}
                      onChange={(value) => setSettings({ ...settings, soundEnabled: value })}
                    />
                  </div>

                  <div className="flex items-center justify-between py-4">
                    <div className="flex items-center gap-3">
                      <Save className="w-5 h-5 text-slate-400" />
                      <div>
                        <p className="text-white font-medium">Auto-save Conversations</p>
                        <p className="text-slate-400 text-sm">Automatically save conversation history</p>
                      </div>
                    </div>
                    <ToggleSwitch
                      enabled={settings.autoSave}
                      onChange={(value) => setSettings({ ...settings, autoSave: value })}
                    />
                  </div>
                </div>
              </div>
            )}

            {/* Notifications Section */}
            {activeSection === "notifications" && (
              <div className="space-y-6">
                <h2 className="text-xl font-semibold text-white mb-6">Notification Settings</h2>

                <div className="space-y-4">
                  <div className="flex items-center justify-between py-4 border-b border-white/10">
                    <div>
                      <p className="text-white font-medium">Email Notifications</p>
                      <p className="text-slate-400 text-sm">Receive updates and tips via email</p>
                    </div>
                    <ToggleSwitch
                      enabled={settings.emailNotifications}
                      onChange={(value) => setSettings({ ...settings, emailNotifications: value })}
                    />
                  </div>

                  <div className="flex items-center justify-between py-4">
                    <div>
                      <p className="text-white font-medium">Push Notifications</p>
                      <p className="text-slate-400 text-sm">Get notified about new features</p>
                    </div>
                    <ToggleSwitch
                      enabled={settings.pushNotifications}
                      onChange={(value) => setSettings({ ...settings, pushNotifications: value })}
                    />
                  </div>
                </div>
              </div>
            )}

            {/* Security Section */}
            {activeSection === "security" && (
              <div className="space-y-6">
                <h2 className="text-xl font-semibold text-white mb-6">Security Settings</h2>

                <div className="space-y-4">
                  <div className="p-4 bg-white/5 rounded-xl">
                    <h3 className="text-white font-medium mb-2">Change Password</h3>
                    <p className="text-slate-400 text-sm mb-4">Update your password regularly for security</p>
                    <button className="px-4 py-2 bg-white/10 rounded-lg text-white text-sm hover:bg-white/20 transition-colors">
                      Update Password
                    </button>
                  </div>

                  <div className="p-4 bg-white/5 rounded-xl">
                    <h3 className="text-white font-medium mb-2">Two-Factor Authentication</h3>
                    <p className="text-slate-400 text-sm mb-4">Add an extra layer of security to your account</p>
                    <button className="px-4 py-2 bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg text-white text-sm hover:shadow-lg hover:shadow-blue-500/25 transition-all">
                      Enable 2FA
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Billing Section */}
            {activeSection === "billing" && (
              <div className="space-y-6">
                <h2 className="text-xl font-semibold text-white mb-6">Billing & Plans</h2>

                <div className="p-6 bg-gradient-to-br from-blue-500/10 to-purple-500/10 border border-blue-500/20 rounded-xl">
                  <div className="flex items-center justify-between mb-4">
                    <div>
                      <p className="text-white font-medium">Pro Plan</p>
                      <p className="text-slate-400 text-sm">$99/month</p>
                    </div>
                    <span className="px-3 py-1 bg-emerald-500/20 text-emerald-400 rounded-full text-sm">Active</span>
                  </div>
                  <p className="text-slate-300 text-sm mb-4">
                    Unlimited conversations, all consultants, 2 languages
                  </p>
                  <button className="px-4 py-2 bg-white/10 rounded-lg text-white text-sm hover:bg-white/20 transition-colors">
                    Manage Subscription
                  </button>
                </div>

                <div className="p-4 bg-white/5 rounded-xl">
                  <h3 className="text-white font-medium mb-2">Payment Method</h3>
                  <p className="text-slate-400 text-sm">Visa ending in 4242</p>
                </div>
              </div>
            )}

            {/* Save Button */}
            <div className="mt-8 pt-6 border-t border-white/10 flex justify-end">
              <button
                onClick={handleSave}
                className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-500 rounded-xl font-medium text-white hover:shadow-lg hover:shadow-blue-500/25 transition-all"
              >
                {saved ? (
                  <>
                    <Check className="w-5 h-5" />
                    Saved!
                  </>
                ) : (
                  <>
                    <Save className="w-5 h-5" />
                    Save Changes
                  </>
                )}
              </button>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
