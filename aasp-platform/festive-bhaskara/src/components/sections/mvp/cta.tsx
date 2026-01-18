'use client';

import { motion } from 'motion/react';
import { ArrowRight, Mail, Calendar, FileText, Github, Linkedin, Twitter, Shield, Sparkles } from 'lucide-react';
import { ctaOptions } from '@/lib/data/mvp-data';

const ctaIcons = {
  'Early Access': Sparkles,
  'Schedule Demo': Calendar,
  'Documentation': FileText,
  'Contact Sales': Mail,
};

export function CTASection() {
  return (
    <div className="space-y-12">
      {/* Hero CTA */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative overflow-hidden rounded-3xl border border-primary/30 bg-gradient-to-br from-primary/20 via-background to-accent/20 p-12"
      >
        {/* Background Effects */}
        <div className="absolute inset-0">
          <div className="absolute top-0 left-1/4 w-96 h-96 bg-primary/20 rounded-full blur-3xl" />
          <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-accent/20 rounded-full blur-3xl" />
        </div>

        <div className="relative text-center max-w-3xl mx-auto">
          <motion.div
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: 1, scale: 1 }}
            className="inline-flex items-center justify-center h-16 w-16 rounded-2xl bg-gradient-to-br from-primary to-accent mb-6"
          >
            <Shield className="h-8 w-8 text-white" />
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="text-4xl md:text-5xl font-bold text-foreground mb-4"
          >
            Ready to Secure Your AI Agents?
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="text-xl text-muted-foreground mb-8"
          >
            Join the companies building trust in the agentic AI era.
            Get early access to the platform that&apos;s defining AI security.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="flex flex-wrap justify-center gap-4"
          >
            <button className="group flex items-center gap-2 rounded-xl bg-primary px-8 py-4 text-lg font-semibold text-primary-foreground transition-all hover:bg-primary/90 hover:shadow-xl hover:shadow-primary/30">
              Request Early Access
              <ArrowRight className="h-5 w-5 transition-transform group-hover:translate-x-1" />
            </button>
            <button className="flex items-center gap-2 rounded-xl border border-border bg-background/50 backdrop-blur-sm px-8 py-4 text-lg font-semibold text-foreground transition-all hover:bg-secondary">
              <Calendar className="h-5 w-5" />
              Book a Demo
            </button>
          </motion.div>

          {/* Trust Indicators */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
            className="flex flex-wrap justify-center items-center gap-6 mt-10 text-sm text-muted-foreground"
          >
            <div className="flex items-center gap-2">
              <Shield className="h-4 w-4 text-green-400" />
              SOC 2 Compliant
            </div>
            <div className="flex items-center gap-2">
              <Shield className="h-4 w-4 text-green-400" />
              GDPR Ready
            </div>
            <div className="flex items-center gap-2">
              <Shield className="h-4 w-4 text-green-400" />
              99.99% Uptime SLA
            </div>
          </motion.div>
        </div>
      </motion.div>

      {/* CTA Options Grid */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
      >
        {ctaOptions.map((option, index) => {
          const Icon = ctaIcons[option.title as keyof typeof ctaIcons] || Mail;

          return (
            <motion.div
              key={option.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.1 * index }}
              className={`group relative overflow-hidden rounded-2xl border p-6 transition-all hover:shadow-lg cursor-pointer ${
                option.primary
                  ? 'border-primary/50 bg-primary/5 hover:bg-primary/10 hover:shadow-primary/20'
                  : 'border-border/50 bg-card/30 hover:bg-card/50 hover:shadow-white/5'
              } backdrop-blur-sm`}
            >
              <div className="absolute top-0 right-0 w-24 h-24 bg-primary/5 rounded-full blur-2xl opacity-0 group-hover:opacity-100 transition-opacity" />

              <div className="relative">
                <div className={`flex h-12 w-12 items-center justify-center rounded-xl mb-4 ${
                  option.primary
                    ? 'bg-primary/20'
                    : 'bg-secondary'
                }`}>
                  <Icon className={`h-6 w-6 ${option.primary ? 'text-primary' : 'text-muted-foreground'}`} />
                </div>

                <h3 className="text-lg font-semibold text-foreground mb-2">{option.title}</h3>
                <p className="text-sm text-muted-foreground mb-4">{option.description}</p>

                <div className="flex items-center gap-2 text-sm font-medium">
                  <span className={option.primary ? 'text-primary' : 'text-foreground'}>
                    {option.action}
                  </span>
                  <ArrowRight className={`h-4 w-4 transition-transform group-hover:translate-x-1 ${
                    option.primary ? 'text-primary' : 'text-foreground'
                  }`} />
                </div>
              </div>
            </motion.div>
          );
        })}
      </motion.div>

      {/* Newsletter Signup */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        className="rounded-2xl border border-border/50 bg-card/30 p-8 backdrop-blur-sm"
      >
        <div className="max-w-2xl mx-auto text-center">
          <Mail className="h-10 w-10 text-primary mx-auto mb-4" />
          <h3 className="text-2xl font-semibold text-foreground mb-2">Stay Updated</h3>
          <p className="text-muted-foreground mb-6">
            Get the latest updates on AI agent security, product releases, and industry insights.
          </p>

          <div className="flex flex-col sm:flex-row gap-3 max-w-md mx-auto">
            <input
              type="email"
              placeholder="Enter your email"
              className="flex-1 rounded-lg border border-border bg-background px-4 py-3 text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
            <button className="rounded-lg bg-primary px-6 py-3 font-medium text-primary-foreground transition-all hover:bg-primary/90">
              Subscribe
            </button>
          </div>

          <p className="text-xs text-muted-foreground mt-4">
            No spam, unsubscribe anytime. Read our privacy policy.
          </p>
        </div>
      </motion.div>

      {/* Contact & Social */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Contact Form */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          className="rounded-2xl border border-border/50 bg-card/30 p-6 backdrop-blur-sm"
        >
          <h3 className="text-xl font-semibold text-foreground mb-6">Get in Touch</h3>
          <form className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <input
                type="text"
                placeholder="First name"
                className="rounded-lg border border-border bg-background px-4 py-2.5 text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
              />
              <input
                type="text"
                placeholder="Last name"
                className="rounded-lg border border-border bg-background px-4 py-2.5 text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
              />
            </div>
            <input
              type="email"
              placeholder="Work email"
              className="w-full rounded-lg border border-border bg-background px-4 py-2.5 text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
            <input
              type="text"
              placeholder="Company"
              className="w-full rounded-lg border border-border bg-background px-4 py-2.5 text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
            <textarea
              placeholder="How can we help?"
              rows={4}
              className="w-full rounded-lg border border-border bg-background px-4 py-2.5 text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50 resize-none"
            />
            <button
              type="submit"
              className="w-full rounded-lg bg-primary py-3 font-medium text-primary-foreground transition-all hover:bg-primary/90"
            >
              Send Message
            </button>
          </form>
        </motion.div>

        {/* Company Info */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          className="space-y-6"
        >
          <div className="rounded-2xl border border-border/50 bg-card/30 p-6 backdrop-blur-sm">
            <h3 className="text-xl font-semibold text-foreground mb-4">Connect With Us</h3>
            <div className="flex gap-4">
              {[
                { icon: Twitter, label: 'Twitter', href: '#' },
                { icon: Linkedin, label: 'LinkedIn', href: '#' },
                { icon: Github, label: 'GitHub', href: '#' },
              ].map(({ icon: Icon, label, href }) => (
                <a
                  key={label}
                  href={href}
                  className="flex h-12 w-12 items-center justify-center rounded-xl bg-secondary hover:bg-secondary/80 transition-colors"
                  aria-label={label}
                >
                  <Icon className="h-5 w-5 text-foreground" />
                </a>
              ))}
            </div>
          </div>

          <div className="rounded-2xl border border-border/50 bg-card/30 p-6 backdrop-blur-sm">
            <h3 className="text-xl font-semibold text-foreground mb-4">Office</h3>
            <address className="not-italic text-muted-foreground space-y-2">
              <p>AASP Inc.</p>
              <p>548 Market St, Suite 95000</p>
              <p>San Francisco, CA 94104</p>
              <p className="pt-2">
                <a href="mailto:hello@aasp.security" className="text-primary hover:underline">
                  hello@aasp.security
                </a>
              </p>
            </address>
          </div>

          <div className="rounded-2xl border border-primary/30 bg-primary/5 p-6">
            <h3 className="text-lg font-semibold text-foreground mb-2">Ready to start?</h3>
            <p className="text-sm text-muted-foreground mb-4">
              Book a 15-minute call with our team to discuss your AI security needs.
            </p>
            <button className="flex items-center gap-2 text-primary font-medium hover:underline">
              <Calendar className="h-4 w-4" />
              Schedule a call
            </button>
          </div>
        </motion.div>
      </div>

      {/* Footer */}
      <motion.div
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: true }}
        className="border-t border-border/30 pt-8 text-center"
      >
        <div className="flex items-center justify-center gap-2 mb-4">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-primary to-accent">
            <Shield className="h-4 w-4 text-white" />
          </div>
          <span className="text-lg font-bold text-foreground">AASP</span>
        </div>
        <p className="text-sm text-muted-foreground mb-4">
          AI Agent Security Platform - Building trust for the agentic AI era
        </p>
        <div className="flex flex-wrap justify-center gap-6 text-sm text-muted-foreground">
          <a href="#" className="hover:text-foreground transition-colors">Privacy Policy</a>
          <a href="#" className="hover:text-foreground transition-colors">Terms of Service</a>
          <a href="#" className="hover:text-foreground transition-colors">Security</a>
          <a href="#" className="hover:text-foreground transition-colors">Status</a>
        </div>
        <p className="text-xs text-muted-foreground mt-6">
          © 2025 AASP Inc. All rights reserved.
        </p>
      </motion.div>
    </div>
  );
}
