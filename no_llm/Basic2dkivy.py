import kivy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty, NumericProperty, BooleanProperty, StringProperty, ListProperty
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Rectangle, Ellipse, Line
from kivy.animation import Animation
from kivy.vector import Vector
from kivy.uix.button import Button
from kivy.uix.settings import SettingsWithSidebar
import random

# Define a visually appealing color scheme
background_color = (0.1, 0.1, 0.2, 1)
player_color = (0.3, 0.6, 1, 1)
npc_color = (1, 0.5, 0.5, 1)
monster_color = (0.6, 0.1, 0.1, 1)
dialogue_box_color = (0.2, 0.2, 0.2, 0.8)
glow_color = (1, 1, 0, 0.5)
info_box_color = (0.1, 0.1, 0.3, 0.9)
button_color = (0.3, 0.3, 0.3, 0.8)

# Item definitions
items = [
    {'name': 'Common Sword', 'type': 'weapon', 'damage': 10, 'rarity': 0.6},
    {'name': 'Rare Sword', 'type': 'weapon', 'damage': 20, 'rarity': 0.3},
    {'name': 'Legendary Sword', 'type': 'weapon', 'damage': 50, 'rarity': 0.1},
    {'name': 'Common Armor', 'type': 'armor', 'defense': 5, 'rarity': 0.6},
    {'name': 'Rare Armor', 'type': 'armor', 'defense': 15, 'rarity': 0.3},
    {'name': 'Legendary Armor', 'type': 'armor', 'defense': 30, 'rarity': 0.1}
]

class Player(Widget):
    velocity = NumericProperty(200)  # pixels per second
    health = NumericProperty(100)
    max_health = NumericProperty(100)
    is_moving = BooleanProperty(False)
    weapon = ObjectProperty(None)
    armor = ObjectProperty(None)
    inventory = ListProperty([])

    def __init__(self, **kwargs):
        super(Player, self).__init__(**kwargs)
        with self.canvas:
            Color(*player_color)
            self.ellipse = Ellipse(pos=self.pos, size=(50, 50))
        self.bind(pos=self.update_ellipse)
        self.destination = None
        self.weapon = {'name': 'Fists', 'damage': 5}
        self.armor = {'name': 'Clothes', 'defense': 1}

    def update_ellipse(self, *args):
        self.ellipse.pos = self.pos

    def move_to(self, destination):
        self.destination = destination
        self.is_moving = True

    def stop_moving(self):
        self.is_moving = False
        self.destination = None

    def update(self, dt):
        if self.is_moving and self.destination:
            direction = Vector(self.destination) - Vector(self.pos)
            if direction.length() <= self.velocity * dt:
                self.pos = self.destination
                self.stop_moving()
            else:
                direction = direction.normalize()
                self.pos = Vector(self.pos) + direction * self.velocity * dt

    def take_damage(self, damage):
        self.health -= damage
        if self.health <= 0:
            self.health = 0
            print("Player has died.")

    def heal(self, amount):
        self.health += amount
        if self.health > self.max_health:
            self.health = self.max_health

    def equip_item(self, item):
        if item['type'] == 'weapon':
            self.weapon = item
        elif item['type'] == 'armor':
            self.armor = item

    def add_to_inventory(self, item):
        self.inventory.append(item)

class NPC(Widget):
    dialogue_index = NumericProperty(0)

    def __init__(self, **kwargs):
        super(NPC, self).__init__(**kwargs)
        with self.canvas:
            Color(*npc_color)
            self.ellipse = Ellipse(pos=self.pos, size=(50, 50))
        self.bind(pos=self.update_ellipse)
        self.dialogue = ["Hello, traveler!", "It's a fine day today.", "Take care on your journey."]

    def update_ellipse(self, *args):
        self.ellipse.pos = self.pos

    def next_dialogue(self):
        self.dialogue_index = (self.dialogue_index + 1) % len(self.dialogue)
        return self.dialogue[self.dialogue_index]

class Monster(Widget):
    health = NumericProperty(50)
    max_health = NumericProperty(50)

    def __init__(self, **kwargs):
        super(Monster, self).__init__(**kwargs)
        with self.canvas:
            Color(*monster_color)
            self.ellipse = Ellipse(pos=self.pos, size=(50, 50))
            self.health_bar = Line(rectangle=(self.x, self.y + 55, 50, 5), width=2)
            self.health_fill = Rectangle(pos=(self.x, self.y + 55), size=(50, 5))
            self.canvas.add(self.health_fill)
            self.canvas.add(self.health_bar)
        self.bind(pos=self.update_ellipse, health=self.update_health_bar)

    def update_ellipse(self, *args):
        self.ellipse.pos = self.pos
        self.health_bar.rectangle = (self.x, self.y + 55, 50, 5)
        self.health_fill.pos = (self.x, self.y + 55)

    def update_health_bar(self, *args):
        health_ratio = self.health / self.max_health
        self.health_fill.size = (50 * health_ratio, 5)
        if health_ratio <= 0:
            self.parent.remove_widget(self)

    def take_damage(self, damage):
        self.health -= damage
        if self.health <= 0:
            self.drop_loot()
            self.parent.remove_widget(self)

    def drop_loot(self):
        drops = []
        for item in items:
            if random.random() < item['rarity']:
                drops.append(item)
        coins = random.randint(1, 100)
        drops.append({'name': 'Coins', 'amount': coins})
        return drops

class DialogueBox(Label):
    def __init__(self, **kwargs):
        super(DialogueBox, self).__init__(**kwargs)
        self.size_hint = (1, None)
        self.height = 100
        self.opacity = 0
        self.text = "Click to continue..."
        with self.canvas.before:
            Color(*dialogue_box_color)
            self.rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(pos=self.update_rect, size=self.update_rect)

    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

    def show(self, dialogue):
        self.text = dialogue
        Animation(opacity=1, duration=0.5).start(self)

    def hide(self):
        Animation(opacity=0, duration=0.5).start(self)

class InfoBox(Label):
    player_pos = StringProperty("Player Position: (0, 0)")
    current_task = StringProperty("Current Task: None")
    player_health = StringProperty("Player Health: 100")
    player_weapon = StringProperty("Weapon: Fists")
    player_armor = StringProperty("Armor: Clothes")
    player_inventory = StringProperty("Inventory: []")

    def __init__(self, **kwargs):
        super(InfoBox, self).__init__(**kwargs)
        self.size_hint = (1, None)
        self.height = 100
        self.text = self.player_pos
        with self.canvas.before:
            Color(*info_box_color)
            self.rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(pos=self.update_rect, size=self.update_rect)

    def update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

    def update_info(self, player):
        self.player_pos = f"Player Position: {player.pos}"
        self.player_health = f"Player Health: {player.health}"
        self.player_weapon = f"Weapon: {player.weapon['name']}"
        self.player_armor = f"Armor: {player.armor['name']}"
        self.player_inventory = f"Inventory: {[item['name'] for item in player.inventory]}"
        self.text = f"{self.player_pos}\n{self.current_task}\n{self.player_health}\n{self.player_weapon}\n{self.player_armor}\n{self.player_inventory}"

    def update_task(self, task):
        self.current_task = f"Current Task: {task}"
        self.text = f"{self.player_pos}\n{self.current_task}\n{self.player_health}\n{self.player_weapon}\n{self.player_armor}\n{self.player_inventory}"

class GameScreen(FloatLayout):
    player = ObjectProperty(None)
    npc = ObjectProperty(None)
    monster = ObjectProperty(None)
    dialogue_box = ObjectProperty(None)
    info_box = ObjectProperty(None)
    settings_button = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(GameScreen, self).__init__(**kwargs)
        with self.canvas.before:
            Color(*background_color)
            self.rect = Rectangle(size=Window.size)
        
        self.player = Player(pos=(100, 100))
        self.npc = NPC(pos=(300, 300))
        self.monster = Monster(pos=(500, 500))
        self.dialogue_box = DialogueBox(pos=(0, 0))
        self.info_box = InfoBox(pos=(0, Window.height - 50))
        self.settings_button = Button(text="Settings", size_hint=(0.1, 0.05), pos_hint={'right': 1, 'top': 1}, background_color=button_color)

        self.add_widget(self.player)
        self.add_widget(self.npc)
        self.add_widget(self.monster)
        self.add_widget(self.dialogue_box)
        self.add_widget(self.info_box)
        self.add_widget(self.settings_button)

        self.settings_button.bind(on_release=self.open_settings)
        self.dialogue_box.bind(on_touch_down=self.on_dialogue_touch)

        Clock.schedule_interval(self.update, 1.0 / 60.0)

        Window.bind(on_key_down=self.on_key_down)

    def update(self, dt):
        self.player.update(dt)
        self.info_box.update_info(self.player)

    def on_touch_down(self, touch, *args):
        if hasattr(touch, 'pos'):
            if self.settings_button.collide_point(*touch.pos):
                self.open_settings(self.settings_button)
                return True
            elif self.player.collide_point(*touch.pos):
                self.dialogue_box.hide()
            elif self.npc.collide_point(*touch.pos):
                self.player.move_to(self.npc.pos)
            elif self.monster.collide_point(*touch.pos):
                self.player.move_to(self.monster.pos)
                if self.player.collide_widget(self.monster):
                    self.attack_monster()
            else:
                self.player.move_to(touch.pos)

    def on_touch_up(self, touch, *args):
        if hasattr(touch, 'pos'):
            if self.player.collide_widget(self.npc):
                self.dialogue_box.show(self.npc.next_dialogue())
                self.info_box.update_task("Talk to the NPC")
            else:
                self.player.stop_moving()

    def on_key_down(self, window, key, *args):
        movement_speed = self.player.velocity / 60
        if key == 273:  # Up arrow
            self.player.move_to((self.player.x, self.player.y + movement_speed))
        elif key == 274:  # Down arrow
            self.player.move_to((self.player.x, self.player.y - movement_speed))
        elif key == 275:  # Right arrow
            self.player.move_to((self.player.x + movement_speed, self.player.y))
        elif key == 276:  # Left arrow
            self.player.move_to((self.player.x - movement_speed, self.player.y))
        elif key == ord('w'):
            self.player.move_to((self.player.x, self.player.y + movement_speed))
        elif key == ord('s'):
            self.player.move_to((self.player.x, self.player.y - movement_speed))
        elif key == ord('d'):
            self.player.move_to((self.player.x + movement_speed, self.player.y))
        elif key == ord('a'):
            self.player.move_to((self.player.x - movement_speed, self.player.y))

    def attack_monster(self):
        self.monster.take_damage(self.player.weapon['damage'])
        if self.monster.health > 0:
            self.player.take_damage(5)  # The monster retaliates
            self.info_box.update_task(f"Fighting monster, Health: {self.monster.health}")
        else:
            loot = self.monster.drop_loot()
            for item in loot:
                if item['name'] == 'Coins':
                    self.info_box.update_task(f"Found {item['amount']} coins!")
                else:
                    self.player.add_to_inventory(item)
                    self.info_box.update_task(f"Found {item['name']}!")

    def open_settings(self, instance):
        app = App.get_running_app()
        app.open_settings()

    def on_dialogue_touch(self, instance, touch):
        if self.dialogue_box.collide_point(*touch.pos):
            self.dialogue_box.hide()

class RPGApp(App):
    use_kivy_settings = False

    def build(self):
        self.settings_cls = SettingsWithSidebar
        game_screen = GameScreen()
        Window.bind(on_touch_down=game_screen.on_touch_down)
        Window.bind(on_touch_up=game_screen.on_touch_up)
        return game_screen

    def build_config(self, config):
        config.setdefaults('game', {
            'player_speed': '200',
        })

    def build_settings(self, settings):
        settings.add_json_panel('Game Settings', self.config, data='''
        [
            {
                "type": "numeric",
                "title": "Player Speed",
                "desc": "Set the speed of the player.",
                "section": "game",
                "key": "player_speed"
            }
        ]''')

    def on_config_change(self, config, section, key, value):
        if key == 'player_speed':
            self.root.player.velocity = int(value)

if __name__ == "__main__":
    RPGApp().run()
