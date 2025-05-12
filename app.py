#!/usr/bin/env python

from data_fetcher import QuestionFetcher
from data_fetcher import DataFetcher

class BayesianMBTIApp:
    def __init__(self):
        self.name: str = "guest"
        self.email: str = "guest"
        self.telp: str = ""

        # All 16 possible MBTI types
        self.mbti_types = [
            "ISTJ", "ISFJ", "INFJ", "INTJ",
            "ISTP", "ISFP", "INFP", "INTP",
            "ESTP", "ESFP", "ENFP", "ENTP",
            "ESTJ", "ESFJ", "ENFJ", "ENTJ"
        ]

        # Initialize equal prior probabilities for each type (1/16)
        self.type_probabilities = {}
        for i in range(0, len(self.mbti_types)):
            self.type_probabilities[self.mbti_types[i]] = 1/16
        # self.type_probabilities = {mbti_type: 1/16 for mbti_type in self.mbti_types}

        # Questions with associated trait likelihoods
        qfetch = QuestionFetcher()
        qfetch.parse("./data/question.csv")
        self.questions = qfetch.question

        jfetch = DataFetcher()
        jfetch.read_file("./data/raw_mbti.csv")
        self.job_data = jfetch.data

        # TODO: Will use the POWER OF AI for this.
        self.personality_descriptions = {
            "ISTJ": "ISTJs are structured, detail-oriented, and grounded in reality. Their introverted sensing (Si) makes them excellent at recalling past data and applying proven methods, ideal for roles like software engineering or systems analysis, where consistency, precision, and clear frameworks are essential. Their introversion aligns with tasks requiring deep, uninterrupted focus, such as debugging or systems optimization, where quiet environments allow them to excel without distractions.",
            "ISFJ": "ISFJs combine empathy with a high degree of responsibility. Their Si-Fe function stack makes them deeply attentive to user needs while valuing thoroughness and tradition. As technical writers, QA testers, or support engineers, they excel by applying proven methods while ensuring others feel supported and guided. Their introversion helps them patiently and carefully address complex problems, often behind the scenes, where their steadiness shines.",
            "INFJ": "INFJs are visionary idealists with strong internal values and future-oriented thinking (Ni-Fe). In roles like UX design, AI ethics, or mission-driven data science, they strive to humanize technology and ensure its use aligns with moral or societal betterment. Their introversion and introspection allow them to identify subtle patterns and imagine long-term impacts, making them uniquely positioned to advocate for ethical innovation and empathetic design.",
            "INTJ": "INTJs are strategic visionaries driven by insight and logic (Ni-Te). Their independent nature thrives in roles like software architecture, machine learning, or startup leadership, where they can build long-term systems from abstract ideas. Their introversion supports long hours of solitary planning, prototyping, and theoretical work—activities that require intense focus and internal vision. They often lead by competence rather than charisma, thriving in meritocratic environments.",
            "ISTP": "ISTPs are pragmatic problem-solvers who love working with real-world systems. Their Ti-Se combination makes them excel at analyzing how things work, often in high-stakes, hands-on roles like embedded systems, DevOps, or cybersecurity. They enjoy autonomy and dislike micromanagement, making introversion ideal for their need to tinker and experiment uninterrupted. Their calm demeanor under pressure makes them reliable in critical debugging or operational tasks.",
            "ISFP": "ISFPs are sensitive creators who care about beauty, user experience, and emotional impact (Fi-Se). In UI/UX design, front-end work, or multimedia coding, they express their inner values through visual storytelling and human-centered interfaces. Their introverted feeling enables them to deeply consider how technology makes users feel, while their sensing helps craft detailed, immersive experiences. They often prefer roles where their creativity speaks louder than their words.",
            "INFP": "INFPs are imaginative idealists who seek meaning in their work (Fi-Ne). They gravitate toward creative or socially impactful roles like game design, ethical AI, or narrative development, where emotional depth and innovation matter. Their introversion supports the long hours needed for coding solo projects or exploring abstract problems, while their values-driven mindset ensures their work resonates with a personal mission or story.",
            "INTP": "INTPs are analytical explorers with a passion for theoretical systems and abstract problem-solving (Ti-Ne). They're naturally drawn to algorithm design, AI research, or open-source contributions where originality and logical rigor are prized. Their introversion fosters a love of solitude and intellectual depth, allowing them to dive deep into complex ideas without needing external validation. They thrive in environments where independent innovation is encouraged.",
            "ESTP": "ESTPs are energetic realists who love fast-paced, tangible challenges (Se-Ti). They excel in field engineering, tech sales, or startup operations, where quick thinking and action drive success. Their extroversion helps them connect with diverse teams and clients, while their sensing makes them adept at responding in real time. They bring a hands-on, entrepreneurial spirit to tech environments that need adaptability and resourcefulness under pressure.",
            "ESFP": "ESFPs are charismatic doers who bring warmth and spontaneity to any team (Se-Fi). They thrive in tech support, digital marketing, or community management, where human interaction and storytelling meet technical tools. Their extroversion makes them ideal ambassadors for tech platforms, while their sensitivity helps them intuitively meet user needs. They shine when technology is a medium for connection, expression, or experience.",
            "ENFP": "ENFPs are enthusiastic innovators driven by vision and human potential (Ne-Fi). In startup teams, product ideation, or innovation labs, they generate bold ideas and inspire collaboration. Their extroversion supports dynamic brainstorming and stakeholder engagement, while their intuition sees connections others might miss. They prefer purpose-driven environments where adaptability and creativity are not just valued, but necessary for growth.",
            "ENTP": "ENTPs are clever and inventive problem-solvers (Ne-Ti), well-suited for roles involving disruption, exploration, or systems reimagination—such as product management, startup founding, or R&D. Their extroversion energizes them in fast-moving discussions and multidisciplinary collaboration, while their curiosity keeps them iterating. They don’t just adapt to change—they cause it, challenging norms and proposing radical alternatives with logic and charm.",
            "ESTJ": "ESTJs are natural organizers who lead with efficiency, structure, and decisiveness (Te-Si). Their strength lies in managing projects, IT departments, or enterprise systems, where clarity and execution are non-negotiable. Their extroversion makes them assertive communicators and reliable decision-makers, while their sensing helps them enforce practical, tested methods. They bring order to chaos and ensure standards are met with consistency.",
            "ESFJ": "ESFJs are nurturing coordinators who prioritize harmony and responsibility (Fe-Si). In customer success, HR tech, or user education, they bring empathy and diligence to every interaction. Their extroversion supports team-building and proactive outreach, while their sensing ensures no detail is overlooked. They excel at translating technical complexity into approachable solutions for end users, helping bridge the gap between people and systems.",
            "ENFJ": "ENFJs are inspirational mentors who guide others with vision and care (Fe-Ni). As edtech leaders, team coaches, or UX researchers, they empower others to grow through technology. Their extroversion allows them to motivate and unify teams, while their intuition helps them anticipate long-term user and team needs. They see technology as a tool for elevating people and ideas, not just solving problems.",
            "ENTJ": "ENTJs are commanding strategists who thrive on leadership and long-term planning (Te-Ni). Ideal as CTOs, tech founders, or lead architects, they turn vision into scalable reality through bold execution. Their extroversion allows them to rally teams and stakeholders, while their intuition fuels innovation. They prefer environments that reward drive, systems thinking, and tangible outcomes, often rising naturally to the top of hierarchical tech structures."
        }

    def set_email(self, email:str):
        self.email = email

    def set_name(self, name:str):
        self.name = name

    def set_telp(self, telp: str):
        self.telp = telp

    def reset(self):
        for mbti_type in self.mbti_types:
            self.type_probabilities[mbti_type] = 1/16
