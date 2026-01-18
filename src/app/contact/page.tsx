"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { motion } from "motion/react";
import {
  ArrowLeft,
  Mail,
  User,
  Building2,
  MessageSquare,
  Send,
  Phone,
  MapPin,
  Clock,
  CheckCircle2,
  Shield
} from "lucide-react";
import { AnimatedAvatarGroup } from "@/components/ui/animated-avatar-group";

// Consultant avatars - same as hero section
const CONSULTANT_AVATARS = [
  { src: "/avatars/manEN.png", fallback: "MR", name: "Marcus Reynolds" },
  { src: "/avatars/womenEN.png", fallback: "SC", name: "Sarah Chen" },
  { src: "/avatars/Man2EN.png", fallback: "DK", name: "David Kim" },
  { src: "/avatars/male1.png", fallback: "JD", name: "Jean Dupont" },
  { src: "/avatars/Women.png", fallback: "CB", name: "Claire Bernard" },
  { src: "/avatars/women2.png", fallback: "ML", name: "Marie Laurent" },
];

// Navbar Component
function Navbar() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 50);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <motion.nav
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled ? 'bg-slate-950/90 backdrop-blur-xl border-b border-white/10' : ''
      }`}
    >
      <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
        <Link href="/" className="text-xl font-bold text-white">
          SF Consultant<span className="text-[#00A1E0]">AI</span>
        </Link>

        <div className="hidden md:flex items-center gap-8">
          <Link href="/#features" className="text-slate-400 hover:text-white transition-colors">Features</Link>
          <Link href="/#pricing" className="text-slate-400 hover:text-white transition-colors">Pricing</Link>
          <Link href="/dashboard/docs" className="text-slate-400 hover:text-white transition-colors">Docs</Link>
        </div>

        <div className="flex items-center gap-4">
          <Link href="/login" className="text-slate-400 hover:text-white transition-colors">
            Log in
          </Link>
          <Link href="/login">
            <button className="px-4 py-2 bg-[#00A1E0] rounded-lg font-medium text-white hover:bg-[#0087be] hover:shadow-lg hover:shadow-[#00A1E0]/25 transition-all duration-300">
              Get Started
            </button>
          </Link>
        </div>
      </div>
    </motion.nav>
  );
}

// Footer Component
function Footer() {
  return (
    <footer className="py-16 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 border-t border-white/10">
      <div className="max-w-6xl mx-auto px-6">
        <div className="grid md:grid-cols-4 gap-12 mb-12">
          <div>
            <h3 className="text-xl font-bold text-white mb-4">SF Consultant AI</h3>
            <p className="text-slate-400">
              AI-powered Salesforce consulting at your fingertips.
            </p>
          </div>
          <div>
            <h4 className="font-semibold text-white mb-4">Product</h4>
            <ul className="space-y-2 text-slate-400">
              <li><Link href="/dashboard/marketplace" className="hover:text-[#00A1E0] transition-colors">Consultants</Link></li>
              <li><Link href="/#pricing" className="hover:text-[#00A1E0] transition-colors">Pricing</Link></li>
              <li><Link href="/dashboard/docs" className="hover:text-[#00A1E0] transition-colors">Documentation</Link></li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold text-white mb-4">Company</h4>
            <ul className="space-y-2 text-slate-400">
              <li><Link href="/about" className="hover:text-[#00A1E0] transition-colors">About</Link></li>
              <li><Link href="/contact" className="hover:text-[#00A1E0] transition-colors">Contact</Link></li>
              <li><Link href="/careers" className="hover:text-[#00A1E0] transition-colors">Careers</Link></li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold text-white mb-4">Legal</h4>
            <ul className="space-y-2 text-slate-400">
              <li><Link href="/privacy" className="hover:text-[#00A1E0] transition-colors">Privacy</Link></li>
              <li><Link href="/terms" className="hover:text-[#00A1E0] transition-colors">Terms</Link></li>
              <li><Link href="/security" className="hover:text-[#00A1E0] transition-colors">Security</Link></li>
            </ul>
          </div>
        </div>

        <div className="pt-8 border-t border-white/10 flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="text-slate-500 text-sm">
            &copy; 2026 SF Consultant AI. All rights reserved.
          </p>
          <div className="flex items-center gap-2 text-slate-500 text-sm">
            <Shield className="w-4 h-4" />
            <span>SOC 2 Compliant</span>
          </div>
        </div>
      </div>
    </footer>
  );
}

export default function ContactPage() {
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    company: "",
    subject: "",
    message: ""
  });
  const [isLoading, setIsLoading] = useState(false);
  const [isSubmitted, setIsSubmitted] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    setFormData(prev => ({
      ...prev,
      [e.target.name]: e.target.value
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    // Simulate form submission
    await new Promise(resolve => setTimeout(resolve, 1500));

    setIsLoading(false);
    setIsSubmitted(true);
  };

  if (isSubmitted) {
    return (
      <div className="min-h-screen bg-slate-950">
        <Navbar />
        <div className="flex items-center justify-center p-8 pt-24 min-h-screen">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
            className="text-center max-w-md"
          >
            <div className="inline-flex p-4 rounded-full bg-green-500/20 mb-6">
              <CheckCircle2 className="w-12 h-12 text-green-500" />
            </div>
            <h1 className="text-3xl font-bold text-white mb-4">Message Sent!</h1>
            <p className="text-slate-400 mb-8">
              Thank you for reaching out. Our team will get back to you within 24 hours.
            </p>
            <Link
              href="/"
              className="inline-flex items-center gap-2 px-6 py-3 bg-slate-800 rounded-xl text-white hover:bg-slate-700 transition-colors"
            >
              <ArrowLeft className="w-4 h-4" />
              Back to home
            </Link>
          </motion.div>
        </div>
        <Footer />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col">
      <Navbar />
      <div className="flex-1 flex pt-16">
      {/* Left Panel - Form */}
      <div className="flex-1 flex items-center justify-center p-8">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5 }}
          className="w-full max-w-lg"
        >
          {/* Back Link */}
          <Link
            href="/"
            className="inline-flex items-center gap-2 text-slate-400 hover:text-white transition-colors mb-8"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to home
          </Link>

          {/* Header */}
          <div className="mb-8">
            <h1 className="text-3xl font-bold text-white mb-2">
              Get in Touch
            </h1>
            <p className="text-slate-400">
              Have questions about SF Consultant AI? We'd love to hear from you.
            </p>
          </div>

          {/* Contact Form */}
          <form onSubmit={handleSubmit} className="space-y-5">
            {/* Name & Email Row */}
            <div className="grid sm:grid-cols-2 gap-4">
              <div>
                <label htmlFor="name" className="block text-sm font-medium text-slate-300 mb-2">
                  Full Name
                </label>
                <div className="relative">
                  <User className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
                  <input
                    id="name"
                    name="name"
                    type="text"
                    value={formData.name}
                    onChange={handleChange}
                    placeholder="John Doe"
                    required
                    className="w-full pl-10 pr-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-slate-500/50 focus:border-slate-500/50 transition-all"
                  />
                </div>
              </div>

              <div>
                <label htmlFor="email" className="block text-sm font-medium text-slate-300 mb-2">
                  Email Address
                </label>
                <div className="relative">
                  <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
                  <input
                    id="email"
                    name="email"
                    type="email"
                    value={formData.email}
                    onChange={handleChange}
                    placeholder="john@company.com"
                    required
                    className="w-full pl-10 pr-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-slate-500/50 focus:border-slate-500/50 transition-all"
                  />
                </div>
              </div>
            </div>

            {/* Company */}
            <div>
              <label htmlFor="company" className="block text-sm font-medium text-slate-300 mb-2">
                Company <span className="text-slate-500">(optional)</span>
              </label>
              <div className="relative">
                <Building2 className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-500" />
                <input
                  id="company"
                  name="company"
                  type="text"
                  value={formData.company}
                  onChange={handleChange}
                  placeholder="Your company name"
                  className="w-full pl-10 pr-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-slate-500/50 focus:border-slate-500/50 transition-all"
                />
              </div>
            </div>

            {/* Subject */}
            <div>
              <label htmlFor="subject" className="block text-sm font-medium text-slate-300 mb-2">
                Subject
              </label>
              <select
                id="subject"
                name="subject"
                value={formData.subject}
                onChange={handleChange}
                required
                className="w-full px-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-slate-500/50 focus:border-slate-500/50 transition-all appearance-none cursor-pointer"
                style={{ backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%2364748b'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M19 9l-7 7-7-7'%3E%3C/path%3E%3C/svg%3E")`, backgroundRepeat: 'no-repeat', backgroundPosition: 'right 12px center', backgroundSize: '20px' }}
              >
                <option value="" className="bg-slate-900">Select a topic</option>
                <option value="general" className="bg-slate-900">General Inquiry</option>
                <option value="sales" className="bg-slate-900">Sales & Pricing</option>
                <option value="enterprise" className="bg-slate-900">Enterprise Solutions</option>
                <option value="support" className="bg-slate-900">Technical Support</option>
                <option value="partnership" className="bg-slate-900">Partnership Opportunities</option>
              </select>
            </div>

            {/* Message */}
            <div>
              <label htmlFor="message" className="block text-sm font-medium text-slate-300 mb-2">
                Message
              </label>
              <div className="relative">
                <MessageSquare className="absolute left-3 top-3 w-5 h-5 text-slate-500" />
                <textarea
                  id="message"
                  name="message"
                  value={formData.message}
                  onChange={handleChange}
                  placeholder="Tell us how we can help you..."
                  required
                  rows={5}
                  className="w-full pl-10 pr-4 py-3 bg-white/5 border border-white/10 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-slate-500/50 focus:border-slate-500/50 transition-all resize-none"
                />
              </div>
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={isLoading}
              className="w-full py-3.5 bg-slate-800 hover:bg-slate-700 rounded-xl font-semibold text-white transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {isLoading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  Sending...
                </span>
              ) : (
                <>
                  <Send className="w-5 h-5" />
                  Send Message
                </>
              )}
            </button>
          </form>

          {/* Privacy Note */}
          <p className="mt-6 text-center text-slate-500 text-sm">
            By submitting this form, you agree to our{" "}
            <Link href="/privacy" className="text-slate-400 hover:text-white transition-colors">
              Privacy Policy
            </Link>
          </p>
        </motion.div>
      </div>

      {/* Right Panel - Contact Info (White Background) */}
      <div className="hidden lg:flex flex-1 items-center justify-center p-8 relative overflow-hidden bg-white">
        {/* Subtle grid pattern */}
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#0f172a08_1px,transparent_1px),linear-gradient(to_bottom,#0f172a08_1px,transparent_1px)] bg-[size:3rem_3rem]" />

        {/* Content */}
        <div className="relative z-10 max-w-md">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            {/* Avatar Group */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="flex flex-col items-center gap-3 mb-8"
            >
              <AnimatedAvatarGroup
                avatars={CONSULTANT_AVATARS}
                maxVisible={6}
                size="lg"
              />
              <span className="text-sm text-slate-500">Our AI consultants are ready to help</span>
            </motion.div>

            <h2 className="text-3xl font-bold text-slate-900 mb-4 text-center">
              Let's Start a Conversation
            </h2>
            <p className="text-slate-600 text-lg leading-relaxed mb-10 text-center">
              Whether you're exploring AI-powered Salesforce consulting or ready to transform your workflow, we're here to help.
            </p>

            {/* Contact Methods */}
            <div className="space-y-5">
              <div className="flex items-start gap-4 p-4 bg-slate-50 rounded-xl">
                <div className="p-3 bg-slate-900 rounded-xl">
                  <Mail className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h3 className="text-slate-900 font-medium mb-1">Email Us</h3>
                  <p className="text-slate-600">contact@sfconsultant.ai</p>
                </div>
              </div>

              <div className="flex items-start gap-4 p-4 bg-slate-50 rounded-xl">
                <div className="p-3 bg-slate-900 rounded-xl">
                  <Phone className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h3 className="text-slate-900 font-medium mb-1">Call Us</h3>
                  <p className="text-slate-600">+1 (555) 123-4567</p>
                </div>
              </div>

              <div className="flex items-start gap-4 p-4 bg-slate-50 rounded-xl">
                <div className="p-3 bg-slate-900 rounded-xl">
                  <MapPin className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h3 className="text-slate-900 font-medium mb-1">Office</h3>
                  <p className="text-slate-600">San Francisco, CA</p>
                </div>
              </div>

              <div className="flex items-start gap-4 p-4 bg-slate-50 rounded-xl">
                <div className="p-3 bg-slate-900 rounded-xl">
                  <Clock className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h3 className="text-slate-900 font-medium mb-1">Response Time</h3>
                  <p className="text-slate-600">Within 24 hours</p>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
      </div>
      <Footer />
    </div>
  );
}
