import 'package:flutter/material.dart';

class AboutSection extends StatelessWidget {
  const AboutSection({super.key});

  @override
  Widget build(BuildContext context) {
    const Color primary = Color(0xFF2E7D32);
    const Color bg = Color(0xFFF1F8E9);

    return Container(
      color: bg,
      padding: const EdgeInsets.symmetric(vertical: 100, horizontal: 40),
      alignment: Alignment.center,
      child: Column(
        children: [
          const Text(
            "About DysTrace",
            style: TextStyle(
              fontSize: 32,
              fontWeight: FontWeight.bold,
              color: primary,
              letterSpacing: 0.5,
            ),
          ),
          const SizedBox(height: 40),
          const SizedBox(
            width: 850,
            child: Text(
              "DysTrace is not a diagnostic tool but a research-oriented platform that provides valuable insights into reading behaviors. "
              "By combining reading accuracy, voice analysis, comprehension tests, and eye-tracking data, "
              "DysTrace allows educators and clinicians to better understand individual differences in literacy development.\n\n"
              "Our mission is to empower early intervention through data — helping children and adults receive the right support before challenges become barriers.",
              textAlign: TextAlign.center,
              style: TextStyle(fontSize: 16.5, height: 1.8, color: Colors.black87),
            ),
          ),

          const SizedBox(height: 80),
          const Text(
            "Project Contributors",
            style: TextStyle(
              fontSize: 26,
              fontWeight: FontWeight.bold,
              color: primary,
              letterSpacing: 0.4,
            ),
          ),
          const SizedBox(height: 40),

          // ───── Creator Cards ─────
          Wrap(
            spacing: 80,
            runSpacing: 40,
            alignment: WrapAlignment.center,
            children: const [
              _MemberCard(
                name: "Jaeyeon Won",
                role: "University of Texas at Austin\nElectrical and Computer Engineering Major",
              ),
              _MemberCard(
                name: "Issac Choi",
                role: "University of Texas at Austin\nComputer Science Major",
              ),
              _MemberCard(
                name: "Sophie Shim",
                role: "University of South California\nComputer Science Major",
              ),
            ],
          ),
        ],
      ),
    );
  }
}

// ─────────────── Member Card Widget ───────────────
class _MemberCard extends StatelessWidget {
  final String name;
  final String role;

  const _MemberCard({
    required this.name,
    required this.role,
  });

  @override
  Widget build(BuildContext context) {
    const Color primary = Color(0xFF2E7D32);

    return Container(
      width: 260,
      padding: const EdgeInsets.symmetric(vertical: 24, horizontal: 16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(14),
        boxShadow: const [
          BoxShadow(
            color: Colors.black12,
            blurRadius: 10,
            offset: Offset(0, 6),
          ),
        ],
      ),
      child: Column(
        children: [
          // 👇 나중에 사진 넣을 때 이 부분 교체
          CircleAvatar(
            radius: 45,
            backgroundColor: primary.withOpacity(0.1),
            child: const Icon(Icons.person, size: 48, color: primary),
          ),
          const SizedBox(height: 14),
          Text(
            name,
            textAlign: TextAlign.center,
            style: const TextStyle(
              fontSize: 17,
              fontWeight: FontWeight.w700,
              color: primary,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            role,
            textAlign: TextAlign.center,
            style: const TextStyle(
              fontSize: 14,
              height: 1.5,
              color: Colors.black87,
            ),
          ),
        ],
      ),
    );
  }
}
